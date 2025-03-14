import cv2
import time
import logging
import threading
from typing import Generator, Dict, Optional, List
import numpy as np
from queue import Queue, Full, Empty

from config.stream_config import StreamConfig, DEFAULT_STREAM_CONFIG
from stream.stream_manager import StreamManager

logger = logging.getLogger(__name__)

class HTTPStreamHandler:
    """HTTP视频流处理器"""
    
    def __init__(self, 
                 stream_manager: StreamManager,
                 config: StreamConfig = None):
        """初始化HTTP流处理器
        
        Args:
            stream_manager: 视频流管理器实例
            config: 流配置参数，如果为None则使用默认配置
        """
        self.stream_manager = stream_manager
        self.config = config or DEFAULT_STREAM_CONFIG
        self.buffer = Queue(maxsize=self.config.buffer_size)
        self.is_running = False
        self.thread = None
        
        # 性能统计
        self.bytes_sent = 0
        self.start_time = 0
        self.bandwidth_kbps = 0
        
    def start(self) -> bool:
        """启动HTTP流处理"""
        if self.is_running:
            return True
            
        if not self.stream_manager.start():
            return False
            
        self.is_running = True
        self.thread = threading.Thread(target=self._buffer_frames, daemon=True)
        self.thread.start()
        
        self.bytes_sent = 0
        self.start_time = time.time()
        
        logger.info("HTTP流处理已启动")
        return True
        
    def stop(self) -> None:
        """停止HTTP流处理"""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        logger.info("HTTP流处理已停止")
        
    def generate_stream(self) -> Generator[bytes, None, None]:
        """生成HTTP流
        
        Yields:
            编码后的MJPEG流数据块
        """
        boundary = b'--frame\r\n'
        content_type = b'Content-Type: image/jpeg\r\n\r\n'
        
        # 开始流处理
        if not self.is_running:
            self.start()
        
        try:
            while self.is_running:
                try:
                    # 从缓冲区获取一帧
                    frame_bytes = self.buffer.get(timeout=1.0)
                    
                    # 组装HTTP响应片段
                    chunk = boundary + content_type + frame_bytes + b'\r\n'
                    
                    # 更新流量统计
                    self.bytes_sent += len(chunk)
                    elapsed = time.time() - self.start_time
                    if elapsed >= 1.0:
                        self.bandwidth_kbps = (self.bytes_sent / 1024) / elapsed
                        # 重置计数器
                        self.bytes_sent = 0
                        self.start_time = time.time()
                        
                        # 自适应质量调整
                        if self.config.adaptive_quality:
                            self._adjust_quality(self.bandwidth_kbps)
                    
                    yield chunk
                    
                except Empty:
                    # 超时，发送保持活动的边界
                    yield boundary
                    continue
                    
                except Exception as e:
                    logger.error(f"生成流时出错: {str(e)}")
                    # 短暂暂停以避免过度日志记录
                    time.sleep(0.1)
                    continue
        
        finally:
            # 确保资源被正确清理
            if self.is_running:
                self.stop()
                
    def _buffer_frames(self) -> None:
        """将处理后的帧缓冲到队列中"""
        target_interval = 1.0 / self.config.max_fps
        last_frame_time = 0
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # 控制帧率
                elapsed = current_time - last_frame_time
                if elapsed < target_interval:
                    time.sleep(target_interval - elapsed)
                    continue
                
                last_frame_time = current_time
                
                # 获取处理后的帧
                processed_frame = self.stream_manager._get_processed_frame()
                if processed_frame is None:
                    continue
                    
                # 根据配置调整大小
                if processed_frame.shape[1] > self.config.max_width:
                    scale = self.config.max_width / processed_frame.shape[1]
                    new_height = int(processed_frame.shape[0] * scale)
                    processed_frame = cv2.resize(
                        processed_frame, 
                        (self.config.max_width, new_height)
                    )
                
                # 添加FPS和带宽信息
                cv2.putText(
                    processed_frame,
                    f"FPS: {self.stream_manager.current_fps:.1f} | BW: {self.bandwidth_kbps:.1f} KB/s",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
                
                # 编码帧
                ret, buffer = cv2.imencode(
                    '.jpg',
                    processed_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.config.quality]
                )
                
                if ret:
                    # 尝试添加到缓冲区，如果已满则丢弃最旧的帧
                    try:
                        self.buffer.put_nowait(buffer.tobytes())
                    except Full:
                        # 队列已满，取出一项后再放入
                        try:
                            self.buffer.get_nowait()
                            self.buffer.put_nowait(buffer.tobytes())
                        except:
                            pass
            
            except Exception as e:
                logger.error(f"缓冲帧时出错: {str(e)}")
                time.sleep(0.1)
    
    def _adjust_quality(self, bandwidth_kbps: float) -> None:
        """根据可用带宽调整质量
        
        Args:
            bandwidth_kbps: 当前带宽 (KB/s)
        """
        # 带宽过低时降低质量和分辨率
        if bandwidth_kbps < 100:  # 低于100KB/s
            if self.config.quality > 50:
                self.config.quality -= 5
            if self.config.max_width > 320:
                self.config.max_width -= 64
                
        # 带宽充足时提高质量和分辨率
        elif bandwidth_kbps > 500:  # 高于500KB/s
            if self.config.quality < 95:
                self.config.quality += 5
            if self.config.max_width < 1280:
                self.config.max_width += 64
        
        # 确保参数在合理范围内
        self.config.quality = max(30, min(95, self.config.quality))
        self.config.max_width = max(320, min(1280, self.config.max_width))
        
    def get_stream_info(self) -> Dict:
        """获取流信息"""
        return {
            'running': self.is_running,
            'bandwidth_kbps': self.bandwidth_kbps,
            'quality': self.config.quality,
            'max_width': self.config.max_width,
            'max_fps': self.config.max_fps,
            'buffer_size': self.buffer.qsize(),
            'buffer_capacity': self.config.buffer_size
        }

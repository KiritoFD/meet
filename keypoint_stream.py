import time
import logging
import threading
import cv2
import numpy as np
from typing import Generator, Dict, Optional

from nvidia.keypoint_receiver import KeypointReceiver
from nvidia.network_simulator import NetworkSimulator
from nvidia.keypoint_compressor import KeypointCompressor
from pose.types import PoseData

logger = logging.getLogger(__name__)

class KeypointStreamHandler:
    """关键点流处理器，提供基于关键点的视频流"""
    
    def __init__(self, receiver: KeypointReceiver = None):
        """初始化流处理器
        
        Args:
            receiver: 关键点接收器，如果为None则创建新的接收器
        """
        self.receiver = receiver or KeypointReceiver()
        self.network_simulator = NetworkSimulator(profile="medium")
        self.keypoint_compressor = KeypointCompressor(precision=2)
        
        self.is_running = False
        self.demo_mode = False  # 是否使用演示模式
        self.demo_thread = None
        
        # 配置
        self.quality = 85  # JPEG质量
        self.max_width = 640  # 最大宽度
        self.target_fps = 30
        
    def set_reference_frame(self, frame: np.ndarray) -> bool:
        """设置参考帧
        
        Args:
            frame: 参考帧图像
            
        Returns:
            设置是否成功
        """
        return self.receiver.set_reference_frame(frame)
        
    def start(self) -> bool:
        """启动流处理
        
        Returns:
            启动是否成功
        """
        if self.is_running:
            return True
            
        # 启动网络模拟器
        if not self.network_simulator.is_running:
            self.network_simulator.start()
            
        # 启动接收器
        if not self.receiver.is_running:
            if not self.receiver.start():
                logger.error("无法启动关键点接收器")
                return False
                
        self.is_running = True
        
        # 如果启用了演示模式，启动演示线程
        if self.demo_mode:
            self.demo_thread = threading.Thread(target=self._demo_loop, daemon=True)
            self.demo_thread.start()
            
        logger.info("关键点流处理器已启动")
        return True
        
    def stop(self) -> None:
        """停止流处理"""
        self.is_running = False
        
        # 停止演示线程
        if self.demo_thread and self.demo_thread.is_alive():
            self.demo_thread.join(timeout=2.0)
            
        # 停止接收器
        if self.receiver.is_running:
            self.receiver.stop()
            
        # 停止网络模拟器
        if self.network_simulator.is_running:
            self.network_simulator.stop()
            
        logger.info("关键点流处理器已停止")
        
    def enable_demo_mode(self, enable: bool = True) -> None:
        """启用或禁用演示模式（自动生成关键点）
        
        Args:
            enable: 是否启用演示模式
        """
        self.demo_mode = enable
        
        # 如果已经在运行且启用了演示模式，但演示线程未启动
        if self.is_running and self.demo_mode and (not self.demo_thread or not self.demo_thread.is_alive()):
            self.demo_thread = threading.Thread(target=self._demo_loop, daemon=True)
            self.demo_thread.start()
            
        logger.info(f"演示模式已{'启用' if enable else '禁用'}")
    
    def process_keypoint_data(self, data: Dict) -> bool:
        """处理关键点数据
        
        Args:
            data: 关键点数据
            
        Returns:
            处理是否成功
        """
        if not self.is_running:
            return False
            
        try:
            # 压缩关键点数据
            pose_data = PoseData(
                keypoints=data.get('keypoints', []),
                timestamp=data.get('timestamp', time.time()),
                confidence=data.get('confidence', 1.0)
            )
            
            compressed_data = self.keypoint_compressor.compress_pose_data(pose_data)
            
            # 序列化
            serialized = self.keypoint_compressor.serialize_for_transmission(compressed_data)
            data_size = len(serialized)
            
            # 模拟网络传输
            transmission_success = self.network_simulator.simulate_send(data_size)
            
            if transmission_success:
                # 传输成功，将数据传递给接收器
                return self.receiver.receive_keypoint_data(compressed_data)
            else:
                # 传输失败，记录丢包
                logger.debug(f"模拟网络传输失败，丢弃数据包 ({data_size} 字节)")
                return False
                
        except Exception as e:
            logger.error(f"处理关键点数据失败: {str(e)}")
            return False
            
    def generate_stream(self) -> Generator[bytes, None, None]:
        """生成视频流
        
        Yields:
            编码后的视频帧
        """
        if not self.start():
            # 如果无法启动，返回一个错误帧
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                error_frame,
                "Error: Could not start keypoint stream",
                (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            
            ret, buffer = cv2.imencode('.jpg', error_frame)
            if ret:
                error_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + error_bytes + b'\r\n')
            return
            
        # 生成视频流
        last_frame_time = 0
        target_interval = 1.0 / self.target_fps
        
        while self.is_running:
            current_time = time.time()
            
            # 控制帧率
            elapsed = current_time - last_frame_time
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
                continue
                
            last_frame_time = current_time
            
            # 获取最新的生成帧
            frame = self.receiver.get_frame()
            
            if frame is None:
                # 如果没有可用帧，使用参考帧
                frame = self.receiver.reference_frame
                if frame is None:
                    # 如果参考帧也不可用，跳过
                    continue
                    
                # 添加提示信息
                cv2.putText(
                    frame,
                    "Waiting for keypoint data...",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
            
            # 调整尺寸（如果需要）
            if frame.shape[1] > self.max_width:
                scale = self.max_width / frame.shape[1]
                new_height = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (self.max_width, new_height))
                
            # 添加网络状态信息
            network_status = self.network_simulator.get_status()
            cv2.putText(
                frame,
                f"Network: {network_status['bandwidth_kbps']}Kbps, "
                f"Loss: {network_status['packet_loss']:.2%}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            
            # 编码帧
            try:
                ret, buffer = cv2.imencode(
                    '.jpg',
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.quality]
                )
                
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                logger.error(f"编码帧失败: {str(e)}")
                
    def _demo_loop(self) -> None:
        """演示循环，自动生成示例关键点"""
        # 用于生成演示关键点的计数器
        counter = 0
        
        # 示例关键点数据
        base_keypoints = [
            {'x': 0.5, 'y': 0.1, 'z': 0, 'confidence': 0.9},  # 头部
            {'x': 0.5, 'y': 0.2, 'z': 0, 'confidence': 0.9},  # 脖子
            {'x': 0.5, 'y': 0.3, 'z': 0, 'confidence': 0.9},  # 胸部
            {'x': 0.4, 'y': 0.2, 'z': 0, 'confidence': 0.9},  # 左肩
            {'x': 0.6, 'y': 0.2, 'z': 0, 'confidence': 0.9},  # 右肩
            {'x': 0.3, 'y': 0.3, 'z': 0, 'confidence': 0.9},  # 左肘
            {'x': 0.7, 'y': 0.3, 'z': 0, 'confidence': 0.9},  # 右肘
            {'x': 0.2, 'y': 0.4, 'z': 0, 'confidence': 0.9},  # 左手
            {'x': 0.8, 'y': 0.4, 'z': 0, 'confidence': 0.9},  # 右手
            {'x': 0.45, 'y': 0.6, 'z': 0, 'confidence': 0.9}, # 左胯
            {'x': 0.55, 'y': 0.6, 'z': 0, 'confidence': 0.9}, # 右胯
            {'x': 0.45, 'y': 0.8, 'z': 0, 'confidence': 0.9}, # 左膝
            {'x': 0.55, 'y': 0.8, 'z': 0, 'confidence': 0.9}, # 右膝
            {'x': 0.45, 'y': 1.0, 'z': 0, 'confidence': 0.9}, # 左脚
            {'x': 0.55, 'y': 1.0, 'z': 0, 'confidence': 0.9}, # 右脚
        ]
        
        while self.is_running and self.demo_mode:
            try:
                # 生成动态关键点
                keypoints = []
                for i, kp in enumerate(base_keypoints):
                    # 添加一些随机运动
                    x_offset = 0.05 * np.sin(counter / 20 + i * 0.5)
                    y_offset = 0.03 * np.sin(counter / 15 + i * 0.7)
                    
                    keypoints.append({
                        'x': kp['x'] + x_offset,
                        'y': kp['y'] + y_offset,
                        'z': kp['z'],
                        'confidence': kp['confidence']
                    })
                    
                # 创建姿态数据
                pose_data = {
                    'keypoints': keypoints,
                    'timestamp': time.time(),
                    'confidence': 0.9
                }
                
                # 处理关键点数据
                self.process_keypoint_data(pose_data)
                
                counter += 1
                
                # 控制生成速率
                time.sleep(1.0 / 15)  # 15fps的生成速率
                
            except Exception as e:
                logger.error(f"演示循环出错: {str(e)}")
                time.sleep(0.1)
    
    def get_status(self) -> Dict:
        """获取流状态
        
        Returns:
            状态信息字典
        """
        receiver_status = self.receiver.get_status() if self.receiver else {}
        network_status = self.network_simulator.get_status() if self.network_simulator else {}
        
        return {
            'running': self.is_running,
            'demo_mode': self.demo_mode,
            'quality': self.quality,
            'max_width': self.max_width,
            'target_fps': self.target_fps,
            'receiver': receiver_status,
            'network': network_status
        }

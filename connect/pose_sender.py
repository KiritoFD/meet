import logging
from typing import Dict, Optional
import time
from collections import deque
import json
import zlib
from .socket_manager import SocketManager

logger = logging.getLogger(__name__)

class PoseSender:
    def __init__(self, socket_manager: SocketManager):
        self.socket_manager = socket_manager
        
        # 性能统计
        self.frame_count = 0
        self.frame_times = deque(maxlen=100)
        self.stats = {
            "fps": 0.0,
            "latency": 0.0,
            "success_rate": 100.0,
            "failed_frames": 0
        }

        # 帧率控制
        self._target_fps = 30
        self._frame_interval = 1.0 / self._target_fps
        self._last_frame_time = time.time()
        self._performance_mode = True  # 默认启用性能模式
        
        # 压缩设置
        self._compression_enabled = False
        self._compression_level = 6
        
        # 队列管理
        self.queue_size = 100
        self._frame_queue = deque(maxlen=100)
        
        # 性能优化
        self._json_encoder = json.JSONEncoder(separators=(',', ':'))
        
    def set_target_fps(self, fps: float):
        """设置目标帧率"""
        if fps <= 0:
            raise ValueError("FPS must be positive")
        self._target_fps = fps
        self._frame_interval = 1.0 / fps
        self._performance_mode = False  # 设置帧率时禁用性能模式
        self._last_frame_time = time.time()  # 重置时间戳
        
    def enable_compression(self, enabled: bool, level: int = 6):
        """启用或禁用压缩"""
        self._compression_enabled = enabled
        if 1 <= level <= 9:
            self._compression_level = level
        else:
            raise ValueError("Compression level must be between 1 and 9")

    def _get_data_size(self, data: dict) -> int:
        """获取数据大小"""
        json_str = self._json_encoder.encode(data)
        if self._compression_enabled:
            return len(zlib.compress(json_str.encode(), self._compression_level))
        return len(json_str.encode())

    def _compress_data(self, data: dict) -> dict:
        """压缩数据"""
        if not self._compression_enabled:
            return data
            
        try:
            json_str = self._json_encoder.encode(data)
            compressed = zlib.compress(json_str.encode(), self._compression_level)
            return {
                "compressed": True,
                "data": compressed
            }
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return data

    def send_pose_data(self, room: str, pose_results: dict, timestamp: float = None) -> bool:
        """发送姿态数据"""
        if not self.socket_manager.connected:
            return False

        if len(self._frame_queue) >= self.queue_size:
            return False

        try:
            # 准备数据
            data = {
                'room': room,
                'pose_data': pose_results,
                'timestamp': timestamp or time.time(),
                'frame_id': self.frame_count
            }

            # 压缩数据
            compressed_data = self._compress_data(data)
            
            # 发送数据
            self.socket_manager.emit('pose_frame', compressed_data)
            
            # 更新队列
            self._frame_queue.append(compressed_data)
            self.frame_count += 1
            
            # 帧率控制
            if not self._performance_mode:
                current_time = time.time()
                elapsed = current_time - self._last_frame_time
                if elapsed < self._frame_interval:
                    time.sleep(self._frame_interval - elapsed)
                self._last_frame_time = current_time + self._frame_interval
            
            return True

        except Exception as e:
            logger.error(f"Send error: {e}")
            return False

    def _update_stats(self, success: bool, start_time: float):
        """更新性能统计"""
        try:
            end_time = time.time()
            frame_time = end_time - start_time
            
            if frame_time > 0:
                self.frame_times.append(frame_time)
                if len(self.frame_times) >= 2:
                    self.stats["fps"] = len(self.frame_times) / sum(self.frame_times)
            
            self.stats["latency"] = frame_time * 1000
            
            if not success:
                self.stats["failed_frames"] += 1
                
            if self.frame_count > 0:
                self.stats["success_rate"] = (
                    (self.frame_count - self.stats["failed_frames"]) / self.frame_count * 100
                )
        except Exception as e:
            logger.error(f"Stats update error: {e}")

    def get_stats(self) -> Dict[str, float]:
        """获取性能统计"""
        return self.stats.copy()

    def set_performance_mode(self, enabled: bool):
        """设置性能模式"""
        self._performance_mode = enabled

import logging
from typing import Dict, Optional
import time
from collections import deque
from .socket_manager import SocketManager

logger = logging.getLogger(__name__)

class PoseSender:
    def __init__(self, socket_manager: SocketManager):
        """初始化发送器
        Args:
            socket_manager: Socket管理器实例
        """
        self.socket_manager = socket_manager
        
        # 性能统计
        self.frame_count = 0
        self.start_time = time.time()
        self.frame_times = deque(maxlen=100)
        self.stats = {
            "fps": 0.0,
            "latency": 0.0,
            "success_rate": 100.0,
            "failed_frames": 0
        }

    def send_pose_data(self, room: str, pose_results: dict, timestamp: float = None) -> bool:
        """发送姿态数据
        Args:
            room: 房间ID
            pose_results: 姿态数据
            timestamp: 时间戳(可选)
        Returns:
            bool: 是否发送成功
        """
        try:
            if not self.socket_manager.connected:
                logger.error("Socket未连接")
                return False

            # 准备发送数据
            data = {
                'pose_data': pose_results,
                'timestamp': timestamp or time.time(),
                'frame_id': self.frame_count
            }

            # 发送数据
            start_time = time.time()
            success = self.socket_manager.emit('pose_frame', data)
            
            # 更新性能统计
            self._update_stats(success, start_time)
            
            return success

        except Exception as e:
            logger.error(f"发送姿态数据失败: {e}")
            self.stats["failed_frames"] += 1
            return False

    def _update_stats(self, success: bool, start_time: float):
        """更新性能统计"""
        self.frame_count += 1
        end_time = time.time()
        
        # 更新帧时间
        frame_time = end_time - start_time
        self.frame_times.append(frame_time)
        
        # 计算FPS
        if len(self.frame_times) >= 2:
            self.stats["fps"] = 1.0 / (sum(self.frame_times) / len(self.frame_times))
        
        # 更新延迟
        self.stats["latency"] = frame_time * 1000  # 转换为毫秒
        
        # 更新成功率
        if not success:
            self.stats["failed_frames"] += 1
        self.stats["success_rate"] = (
            (self.frame_count - self.stats["failed_frames"]) / self.frame_count * 100
        )

    def get_stats(self) -> Dict[str, float]:
        """获取性能统计
        Returns:
            Dict[str, float]: 性能统计数据
        """
        return self.stats.copy()

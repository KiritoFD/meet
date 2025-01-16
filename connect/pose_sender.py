from typing import Dict, Optional
import logging
from collections import deque
import time
import psutil
from .socket_manager import SocketManager
from .pose_protocol import MediaProtocol, MediaData
from .room_manager import RoomManager

class SendConfig:
    def __init__(self):
        self.target_fps: int = 30
        self.max_queue_size: int = 100
        self.compression_level: int = 6
        self.priority_mode: str = 'realtime'

class PoseSender:
    def __init__(self, room_manager: RoomManager, protocol: MediaProtocol):
        """初始化发送器"""
        self.room_manager = room_manager
        self.protocol = protocol
        self.logger = logging.getLogger(__name__)
        
        # 性能统计
        self.frame_count = 0
        self.start_time = time.time()
        self.frame_times = deque(maxlen=100)
        self.stats = {
            "fps": 0.0,
            "latency": 0.0,
            "success_rate": 100.0,
            "failed_frames": 0,
            "cpu_usage": 0.0,
            "memory_usage": 0.0
        }
        
        # 发送控制
        self.send_config = SendConfig()
        self.is_sending = True
        self.monitoring = False

    async def send_pose_data(self, 
                         pose_results,
                         face_results=None, 
                         hands_results=None,
                         room=None,
                         **kwargs) -> bool:
        """
        兼容旧版本的发送方法
        将调用重定向到新的 send_frame 方法
        """
        self.logger.warning("使用已废弃的 send_pose_data 方法，请使用 send_frame 替代")
        return await self.send_frame(
            room_id=room,
            pose_results=pose_results,
            face_results=face_results,
            hands_results=hands_results
        )

    async def send_frame(self, 
                      room_id: str,
                      pose_results,
                      face_results=None, 
                      hands_results=None) -> bool:
        """发送单帧数据"""
        if not self.is_sending:
            return False

        try:
            start_time = time.time()
            
            # 创建姿态数据
            pose_data = {
                'pose_landmarks': pose_results,
                'face_landmarks': face_results,
                'hand_landmarks': hands_results,
                'timestamp': start_time,
                'frame_id': self.frame_count,
                'data_type': "pose"
            }

            # 如果有 protocol，使用 protocol 处理数据
            if hasattr(self, 'protocol') and self.protocol:
                if not self.protocol.validate(pose_data):
                    self.logger.error("Invalid pose data")
                    return False
                encoded_data = self.protocol.encode(pose_data)
            else:
                encoded_data = pose_data

            # 发送数据
            if room_id:
                success = await self.room_manager.broadcast('pose_data', encoded_data, room_id)
            else:
                # 如果没有指定房间，使用默认的发送方式
                success = await self.socket_manager.emit("pose_frame", encoded_data)

            # 更新统计信息
            self._update_stats(success, start_time)
            
            return success

        except Exception as e:
            self.logger.error(f"Error sending frame: {str(e)}")
            self.stats["failed_frames"] += 1
            return False

    def get_stats(self) -> Dict[str, float]:
        """获取性能统计"""
        self.stats["cpu_usage"] = psutil.cpu_percent()
        self.stats["memory_usage"] = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        return self.stats.copy()

    @property
    def fps(self) -> float:
        """获取当前帧率"""
        return self.stats["fps"]

    def start_monitoring(self):
        """开始性能监控"""
        self.monitoring = True

    def stop_monitoring(self):
        """停止性能监控"""
        self.monitoring = False

    def set_send_config(self, config: SendConfig):
        """设置发送配置"""
        self.send_config = config

    def get_queue_status(self) -> Dict[str, int]:
        """获取队列状态"""
        return {
            "queue_size": len(self.frame_times),
            "max_queue_size": self.send_config.max_queue_size
        }

    def clear_queue(self):
        """清空发送队列"""
        self.frame_times.clear()

    def pause_sending(self):
        """暂停发送"""
        self.is_sending = False

    def resume_sending(self):
        """恢复发送"""
        self.is_sending = True

    def _update_stats(self, success: bool, start_time: float):
        """更新性能统计"""
        self.frame_count += 1
        end_time = time.time()
        
        # 更新FPS
        self.frame_times.append(end_time - start_time)
        if len(self.frame_times) >= 2:
            self.stats["fps"] = 1.0 / (sum(self.frame_times) / len(self.frame_times))

        # 更新延迟
        self.stats["latency"] = (end_time - start_time) * 1000  # 转换为毫秒

        # 更新成功率
        if not success:
            self.stats["failed_frames"] += 1
        self.stats["success_rate"] = ((self.frame_count - self.stats["failed_frames"]) 
                                    / self.frame_count * 100)

        # 性能监控告警
        if self.monitoring:
            self._check_alerts()

    def _check_alerts(self):
        """检查性能告警"""
        if self.stats["fps"] < 20:
            self.logger.warning("帧率过低警告: < 20fps")
        if self.stats["latency"] > 100:
            self.logger.warning("延迟过高警告: > 100ms")
        if self.stats["cpu_usage"] > 50:
            self.logger.warning("CPU使用率过高警告: > 50%") 
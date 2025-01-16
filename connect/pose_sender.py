from typing import Dict, Optional
import logging
from collections import deque
import time
from typing import Optional
import numpy as np
import logging
from .socket_manager import SocketManager
from .pose_protocol import PoseProtocol, PoseData

# 配置logger
logger = logging.getLogger(__name__)

class PoseSender:
    def __init__(self, socketio, socket_manager: SocketManager):
        """初始化发送器
        Args:
            socketio: Socket.IO客户端实例
            socket_manager: Socket管理器实例
        """
        self.socketio = socketio
        self.socket = socket_manager
        self.last_pose = None
        self.change_threshold = 0.005
        
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
        
    def send_pose_data(self, room: str, pose_results, face_results=None, hands_results=None, timestamp=None):
        """发送姿态数据"""
        try:
            # 暂时只打印调试级别的日志
            logger.debug("检测到姿态数据")
            return True
            
        except Exception as e:
            logger.error(f"发送姿态数据失败: {e}")
            return False

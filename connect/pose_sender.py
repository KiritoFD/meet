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
        
    def _has_significant_change(self, new_pose: PoseData) -> bool:
        """检查姿态是否有显著变化"""
        if not self.last_pose or not new_pose.pose_landmarks:
            return True
            
        changes = []
        for old, new in zip(self.last_pose.pose_landmarks, new_pose.pose_landmarks):
            change = abs(old['x'] - new['x']) + abs(old['y'] - new['y'])
            changes.append(change)
            
        return np.mean(changes) > self.change_threshold
        
    def send_pose_data(self, room: str, pose_results, face_results=None, hands_results=None, timestamp=None):
        """发送姿态数据
        Args:
            room: 房间ID
            pose_results: 姿态检测结果
            face_results: 面部检测结果(可选)
            hands_results: 手部检测结果(可选) 
            timestamp: 时间戳(可选)
        """
        try:
            # 编码数据
            pose_data = PoseData(
                pose_landmarks=self.protocol.encode_landmarks(pose_results),
                face_landmarks=self.protocol.encode_landmarks(face_results),
                hand_landmarks=self.protocol.encode_landmarks(hands_results),
                timestamp=timestamp or time.time()
            )
            
            # 压缩并发送
            compressed = self.protocol.compress_data(pose_data)
            return self.socket.emit('pose_data', {'data': compressed})
            
        except Exception as e:
            logger.error(f"发送姿态数据失败: {e}")
            return False 
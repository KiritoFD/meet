import time
from typing import Optional
import numpy as np

class PoseSender:
    def __init__(self, socket_manager: SocketManager, protocol: PoseProtocol):
        self.socket = socket_manager
        self.protocol = protocol
        self.last_pose: Optional[PoseData] = None
        self.change_threshold = 0.005  # 变化阈值
        
    def _has_significant_change(self, new_pose: PoseData) -> bool:
        """检查姿态是否有显著变化"""
        if not self.last_pose or not new_pose.pose_landmarks:
            return True
            
        changes = []
        for old, new in zip(self.last_pose.pose_landmarks, new_pose.pose_landmarks):
            change = abs(old['x'] - new['x']) + abs(old['y'] - new['y'])
            changes.append(change)
            
        return np.mean(changes) > self.change_threshold
        
    def send_pose(self, pose_results, face_results=None, hands_results=None):
        """发送姿态数据"""
        # 1. 编码数据
        pose_data = PoseData(
            pose_landmarks=self.protocol.encode_landmarks(pose_results),
            face_landmarks=self.protocol.encode_landmarks(face_results),
            hand_landmarks=self.protocol.encode_landmarks(hands_results),
            timestamp=time.time()
        )
        
        # 2. 检查变化
        if not self._has_significant_change(pose_data):
            return False
            
        # 3. 压缩并发送
        compressed = self.protocol.compress_data(pose_data)
        success = self.socket.emit('pose_data', {'data': compressed})
        
        if success:
            self.last_pose = pose_data
        return success 
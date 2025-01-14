import json
import zlib
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class PoseData:
    pose_landmarks: Optional[List[Dict[str, float]]] = None
    face_landmarks: Optional[List[Dict[str, float]]] = None
    left_hand_landmarks: Optional[List[Dict[str, float]]] = None
    right_hand_landmarks: Optional[List[Dict[str, float]]] = None
    timestamp: float = 0.0

class PoseSender:
    def __init__(self, socketio, room_manager):
        self.socketio = socketio
        self.room_manager = room_manager
        self.last_pose = None
        self.change_threshold = 0.005  # 变化阈值
        
    def _convert_landmarks_to_dict(self, landmarks) -> List[Dict[str, float]]:
        """将MediaPipe landmarks转换为可序列化的字典列表"""
        if not landmarks:
            return None
        return [
            {
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility if hasattr(landmark, 'visibility') else 1.0
            }
            for landmark in landmarks.landmark
        ]
    
    def _compress_data(self, data: dict) -> bytes:
        """压缩姿态数据"""
        json_str = json.dumps(data)
        return zlib.compress(json_str.encode())
    
    def _has_significant_change(self, new_pose: PoseData) -> bool:
        """检查姿态是否有显著变化"""
        if not self.last_pose or not new_pose.pose_landmarks:
            return True
            
        if not self.last_pose.pose_landmarks:
            return True
            
        # 计算关键点的平均变化
        changes = []
        for old, new in zip(self.last_pose.pose_landmarks, new_pose.pose_landmarks):
            change = abs(old['x'] - new['x']) + abs(old['y'] - new['y']) + abs(old['z'] - new['z'])
            changes.append(change)
            
        avg_change = np.mean(changes)
        return avg_change > self.change_threshold
    
    def send_pose_data(self, room: str, pose_results, face_results, hands_results, timestamp: float):
        """发送姿态数据到指定房间"""
        try:
            # 检查房间是否存在
            if not self.room_manager.room_exists(room):
                return
            
            # 构建姿态数据
            pose_data = PoseData(
                pose_landmarks=self._convert_landmarks_to_dict(pose_results.pose_landmarks) if pose_results else None,
                face_landmarks=self._convert_landmarks_to_dict(face_results.multi_face_landmarks[0]) if face_results and face_results.multi_face_landmarks else None,
                left_hand_landmarks=None,
                right_hand_landmarks=None,
                timestamp=timestamp
            )
            
            # 处理手部数据
            if hands_results and hands_results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                    if idx == 0:
                        pose_data.left_hand_landmarks = self._convert_landmarks_to_dict(hand_landmarks)
                    elif idx == 1:
                        pose_data.right_hand_landmarks = self._convert_landmarks_to_dict(hand_landmarks)
            
            # 检查是否需要发送
            if self._has_significant_change(pose_data):
                # 转换为字典
                data_dict = {
                    'pose': pose_data.pose_landmarks,
                    'face': pose_data.face_landmarks,
                    'left_hand': pose_data.left_hand_landmarks,
                    'right_hand': pose_data.right_hand_landmarks,
                    'timestamp': pose_data.timestamp
                }
                
                # 压缩数据
                compressed_data = self._compress_data(data_dict)
                
                # 发送到房间
                self.socketio.emit('pose_data', 
                                 {'data': compressed_data},
                                 room=room)
                
                # 更新上一帧数据
                self.last_pose = pose_data
                
        except Exception as e:
            print(f"发送姿态数据错误: {str(e)}") 
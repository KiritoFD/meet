import numpy as np
import json
import zlib
import base64
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional

# 配置logger
logger = logging.getLogger(__name__)

@dataclass
class PoseData:
    """姿态数据结构"""
    pose_landmarks: Optional[List[Dict[str, float]]] = None
    face_landmarks: Optional[List[Dict[str, float]]] = None
    hand_landmarks: Optional[List[Dict[str, float]]] = None
    timestamp: float = 0.0

class PoseProtocol:
    def __init__(self, compression_level: int = 6):
        self.compression_level = compression_level
        
    def encode_landmarks(self, landmarks) -> List[Dict[str, float]]:
        """编码关键点数据"""
        if not landmarks:
            return None
        return [
            {
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': getattr(landmark, 'visibility', 1.0)
            }
            for landmark in landmarks.landmark
        ]
        
    def compress_data(self, data: PoseData) -> bytes:
        """压缩姿态数据"""
        json_str = json.dumps({
            'pose': data.pose_landmarks,
            'face': data.face_landmarks,
            'hands': data.hand_landmarks,
            'timestamp': data.timestamp
        })
        return zlib.compress(json_str.encode(), self.compression_level)
        
    def decompress_data(self, compressed: bytes) -> PoseData:
        """解压姿态数据"""
        try:
            json_str = zlib.decompress(compressed).decode()
            data = json.loads(json_str)
            return PoseData(
                pose_landmarks=data['pose'],
                face_landmarks=data['face'],
                hand_landmarks=data['hands'],
                timestamp=data['timestamp']
            )
        except Exception as e:
            logger.error(f"解压数据失败: {e}")
            return None 
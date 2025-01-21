import mediapipe as mp
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class FaceLandmark:
    """面部特征点"""
    x: float
    y: float
    z: float
    visibility: float = 1.0
    
@dataclass
class FaceMesh:
    """面部网格"""
    landmarks: List[FaceLandmark]
    connections: List[tuple]

class FaceDetector:
    """面部检测器"""
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 定义关键特征点索引
        self.FACE_INDICES = {
            'left_eye': [33, 133, 160, 159, 158, 144, 145, 153],
            'right_eye': [362, 263, 387, 386, 385, 373, 374, 380],
            'mouth': [61, 291, 0, 17, 39, 269, 270, 409],
            'nose': [1, 2, 98, 327],
            'left_brow': [70, 63, 105, 66, 107],
            'right_brow': [300, 293, 334, 296, 336]
        }
        
    def detect_landmarks(self, image) -> Optional[List[FaceLandmark]]:
        """检测面部特征点
        
        Args:
            image: RGB格式的图像数组
            
        Returns:
            检测到的特征点列表，如果未检测到则返回None
        """
        results = self.face_mesh.process(image)
        if not results.multi_face_landmarks:
            return None
            
        landmarks = []
        for point in results.multi_face_landmarks[0].landmark:
            landmarks.append(FaceLandmark(
                x=point.x,
                y=point.y,
                z=point.z,
                visibility=1.0  # MediaPipe不提供visibility，默认为1
            ))
            
        return landmarks
        
    def get_face_mesh(self, landmarks: List[FaceLandmark]) -> FaceMesh:
        """获取面部网格
        
        Args:
            landmarks: 特征点列表
            
        Returns:
            面部网格对象
        """
        # MediaPipe Face Mesh的连接关系
        connections = mp.solutions.face_mesh.FACEMESH_TESSELATION
        
        return FaceMesh(
            landmarks=landmarks,
            connections=connections
        )
        
    def get_feature_points(self, landmarks: List[FaceLandmark]) -> Dict[str, List[FaceLandmark]]:
        """获取特定面部特征的点集
        
        Args:
            landmarks: 特征点列表
            
        Returns:
            特征点分组字典
        """
        features = {}
        for name, indices in self.FACE_INDICES.items():
            features[name] = [landmarks[i] for i in indices]
        return features 
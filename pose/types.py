from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import time
import numpy as np

@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float = 1.0
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典格式"""
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'visibility': self.visibility
        }

@dataclass
class BindingPoint:
    """变形控制点"""
    x: float
    y: float
    weight: float = 1.0
    
    def to_array(self) -> np.ndarray:
        """转换为数组格式"""
        return np.array([self.x, self.y])

@dataclass
class DeformRegion:
    """变形区域"""
    name: str
    points: List[BindingPoint]
    center: Optional[Tuple[float, float]] = None
    radius: float = 0.0
    
    def __post_init__(self):
        """初始化后处理"""
        if self.center is None:
            # 计算区域中心
            points = np.array([p.to_array() for p in self.points])
            self.center = tuple(points.mean(axis=0))
            # 计算区域半径
            distances = np.linalg.norm(points - self.center, axis=1)
            self.radius = distances.max()

@dataclass
class PoseData:
    """姿态数据类"""
    landmarks: List[Dict[str, float]]  # 身体关键点列表
    timestamp: float = None  # 时间戳
    confidence: float = 1.0  # 整体置信度
    face_landmarks: Optional[List[Dict[str, float]]] = None  # 面部关键点列表
    hand_landmarks: Optional[List[Dict[str, float]]] = None  # 手部关键点列表
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def get_landmark(self, idx: int) -> Optional[Dict[str, float]]:
        """获取指定索引的关键点"""
        try:
            return self.landmarks[idx]
        except IndexError:
            return None
    
    def is_valid(self, min_confidence: float = 0.5) -> bool:
        """检查姿态数据是否有效"""
        return (self.confidence >= min_confidence and 
                len(self.landmarks) > 0 and 
                all(lm['visibility'] >= min_confidence for lm in self.landmarks))
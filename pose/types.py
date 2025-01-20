from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import time
import numpy as np

@dataclass
class Landmark:
    x: float
    y: float
    z: float = 0.0
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
    landmarks: List[Landmark]
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    
    def __post_init__(self):
        """初始化后处理"""
        self._values = None
    
    @property
    def values(self) -> np.ndarray:
        """获取关键点坐标数组"""
        if self._values is None:
            self._values = np.array([
                [lm.x, lm.y, lm.z] for lm in self.landmarks
            ])
        return self._values
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'landmarks': [lm.to_dict() for lm in self.landmarks],
            'timestamp': self.timestamp,
            'confidence': self.confidence
        } 
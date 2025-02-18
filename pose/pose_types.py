from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import time
import numpy as np

@dataclass
class Landmark:
    """关键点数据类"""
    x: float
    y: float
    z: float
    visibility: float = 1.0

@dataclass
class BindingPoint:
    """绑定点数据类"""
    landmark_index: int
    local_coords: np.ndarray
    weight: float = 1.0

@dataclass
class DeformRegion:
    """变形区域数据类"""
    name: str
    center: np.ndarray
    binding_points: List[BindingPoint]
    mask: Optional[np.ndarray]
    type: str = 'body'  # 添加类型字段：'body', 'face', 'limb'
    
    def __post_init__(self):
        """验证并处理初始数据"""
        # 确保center是numpy数组
        if not isinstance(self.center, np.ndarray):
            self.center = np.array(self.center, dtype=np.float32)
            
        # 确保binding_points是列表
        if self.binding_points is None:
            self.binding_points = []
            
        # 确保mask是正确的类型
        if self.mask is not None and not isinstance(self.mask, np.ndarray):
            raise TypeError("mask must be numpy.ndarray")

    def __getitem__(self, key):
        """支持字典式访问"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{self.__class__.__name__}' has no attribute '{key}'")

@dataclass
class PoseData:
    """姿态数据类"""
    landmarks: List[Dict[str, float]]  # 身体关键点
    timestamp: float = None
    confidence: float = 1.0
    face_landmarks: Optional[List[Dict[str, float]]] = None
    hand_landmarks: Optional[List[Dict[str, float]]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.binding_points is None:
            self.binding_points = []

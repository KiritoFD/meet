from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float

@dataclass
class PoseData:
    landmarks: List[Landmark]
    timestamp: float
    confidence: float

@dataclass
class BindingPoint:
    """绑定点信息"""
    landmark_index: int  # 对应的关键点索引
    local_coords: np.ndarray  # 相对于区域中心的局部坐标
    weight: float  # 影响权重

@dataclass
class DeformRegion:
    """变形区域"""
    center: np.ndarray  # 区域中心坐标
    binding_points: List[BindingPoint]  # 绑定点列表
    mask: Optional[np.ndarray] = None  # 区域蒙版 
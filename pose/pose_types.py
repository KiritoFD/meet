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
    name: str = ''
    center: Optional[Tuple[float, float]] = None
    binding_points: List[BindingPoint] = None
    mask: Optional[np.ndarray] = None

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

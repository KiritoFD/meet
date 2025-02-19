from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import time
import numpy as np

@dataclass
class Landmark:
    """姿态关键点"""
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
    """变形绑定点"""
    landmark_index: int  # 对应的关键点索引
    local_coords: np.ndarray  # 相对于区域中心的局部坐标
    weight: float  # 权重值

@dataclass
class DeformRegion:
    """变形区域"""
    name: str  # 区域名称
    center: np.ndarray  # 区域中心点
    binding_points: List[BindingPoint]  # 绑定点列表
    mask: np.ndarray  # 区域蒙版
    type: str  # 区域类型（'body' 或 'face'）

@dataclass
class PoseData:
    """姿态数据"""
    landmarks: List[Landmark]  # 关键点列表
    timestamp: float = time.time()  # 时间戳
    confidence: float = 1.0  # 置信度
    face_landmarks: Optional[List[Landmark]] = None  # 面部关键点
    hand_landmarks: Optional[List[Landmark]] = None  # 手部关键点
    
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
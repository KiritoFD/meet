from dataclasses import dataclass
from typing import List, Dict, Optional, Union
import numpy as np

@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float = 1.0
    
    def __getitem__(self, key: str) -> float:
        """支持字典式访问,保持向后兼容"""
        if key == 'x':
            return self.x
        elif key == 'y':
            return self.y
        elif key == 'z':
            return self.z
        elif key == 'visibility':
            return self.visibility
        raise KeyError(f"Invalid key: {key}")

@dataclass
class PoseData:
    landmarks: List[Landmark]
    timestamp: float
    confidence: float = 1.0
    
    def validate(self) -> bool:
        """验证姿态数据的有效性"""
        if not self.landmarks:
            return False
            
        # 检查关键点可见度
        visible_landmarks = sum(lm.visibility > 0.5 for lm in self.landmarks)
        if visible_landmarks < len(self.landmarks) * 0.5:  # 至少50%关键点可见
            return False
            
        # 检查置信度
        if self.confidence < 0.3:  # 最小置信度阈值
            return False
            
        return True

@dataclass
class BindingPoint:
    """绑定点信息"""
    landmark_index: int  # 对应的关键点索引
    local_coords: np.ndarray  # 相对于区域中心的局部坐标
    weight: float = 1.0  # 影响权重
    
    def __post_init__(self):
        """验证数据"""
        if not isinstance(self.local_coords, np.ndarray):
            self.local_coords = np.array(self.local_coords)
        if self.local_coords.shape != (2,):
            raise ValueError("local_coords must be 2D coordinates")

@dataclass
class DeformRegion:
    """变形区域"""
    center: np.ndarray  # 区域中心坐标
    binding_points: List[BindingPoint]  # 绑定点列表
    mask: Optional[np.ndarray] = None  # 区域蒙版
    
    def __post_init__(self):
        """验证并初始化数据"""
        if not isinstance(self.center, np.ndarray):
            self.center = np.array(self.center)
        if self.center.shape != (2,):
            raise ValueError("center must be 2D coordinates")
        
        # 确保mask是二维数组
        if self.mask is not None and len(self.mask.shape) != 2:
            raise ValueError("mask must be 2D array") 
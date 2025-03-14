import time
from dataclasses import dataclass, field
from typing import Tuple,List, Dict, Optional, Union, Any
import numpy as np

@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float
    
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
    landmarks: List[Dict[str, float]]
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
                [lm['x'], lm['y'], lm['z']] for lm in self.landmarks
            ])
        return self._values
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'landmarks': self.landmarks,
            'timestamp': self.timestamp,
            'confidence': self.confidence
        }

@dataclass
class Keypoint:
    """
    表示一个关键点的数据结构
    """
    x: float  # 归一化的 x 坐标 (0.0 - 1.0)
    y: float  # 归一化的 y 坐标 (0.0 - 1.0)
    z: float = 0.0  # 归一化的深度 z 坐标 (可选)
    visibility: float = 1.0  # 可见度 (0.0 - 1.0)
    
    def to_dict(self) -> Dict[str, float]:
        """将关键点转换为字典"""
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'visibility': self.visibility
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'Keypoint':
        """从字典创建关键点"""
        return cls(
            x=data.get('x', 0.0),
            y=data.get('y', 0.0),
            z=data.get('z', 0.0),
            visibility=data.get('visibility', 1.0)
        )

@dataclass
class PoseData:
    """
    表示一组姿态关键点数据的结构
    """
    keypoints: List[Keypoint]  # 关键点列表，与NVIDIA兼容
    landmarks: List[Keypoint] = None  # 与NVIDIA模型兼容的别名
    timestamp: float = field(default_factory=time.time)  # 时间戳
    confidence: float = 1.0  # 整体置信度
    
    def __post_init__(self):
        """初始化后处理"""
        # 如果没有设置landmarks，就用keypoints来设置
        if self.landmarks is None and self.keypoints is not None:
            self.landmarks = self.keypoints
        # 反之亦然
        elif self.keypoints is None and self.landmarks is not None:
            self.keypoints = self.landmarks
        # 如果两者都为None，初始化为空列表
        elif self.keypoints is None and self.landmarks is None:
            self.keypoints = []
            self.landmarks = []
    
    def to_dict(self) -> Dict[str, Any]:
        """将姿态数据转换为字典，包括NVIDIA格式兼容的字段"""
        return {
            'keypoints': [kp.to_dict() for kp in self.keypoints],
            'landmarks': [kp.to_dict() for kp in self.keypoints],  # 保持两个字段同步
            'timestamp': self.timestamp,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PoseData':
        """从字典创建姿态数据，支持NVIDIA格式"""
        keypoints = data.get('keypoints')
        landmarks = data.get('landmarks')
        
        # 优先使用keypoints，如果没有则使用landmarks
        points_list = keypoints if keypoints is not None else landmarks
        
        if points_list is not None:
            points = [Keypoint.from_dict(kp) if isinstance(kp, dict) else kp for kp in points_list]
        else:
            points = []
            
        return cls(
            keypoints=points,
            timestamp=data.get('timestamp', time.time()),
            confidence=data.get('confidence', 1.0)
        )
    
    def get_keypoint(self, index: int) -> Optional[Keypoint]:
        """获取特定索引的关键点"""
        if 0 <= index < len(self.keypoints):
            return self.keypoints[index]
        return None
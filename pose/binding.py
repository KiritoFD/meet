from typing import List, Dict, Optional, Any
import numpy as np
from dataclasses import dataclass
from .render_core import ModelRenderer

@dataclass
class Bone:
    start_idx: int
    end_idx: int
    children: List[int]
    influence_radius: float
    transform_matrix: np.ndarray = None  # 新增：用于3D渲染的变换矩阵

@dataclass
class SkeletonBinding:
    reference_frame: np.ndarray
    landmarks: List[Dict[str, float]]
    bones: List[Bone]
    weights: np.ndarray
    mesh_points: np.ndarray
    valid: bool = False
    renderer: Optional[ModelRenderer] = None
    model_matrix: np.ndarray = None  # 新增：3D模型变换矩阵

class PoseBinding:
    def __init__(self, config: Dict[str, Any]):
        """初始化绑定器"""
        self.config = config
    
    def create_binding(self, frame: np.ndarray, 
                      landmarks: List[Dict[str, float]]) -> SkeletonBinding:
        """创建骨骼绑定"""
        # 简化的实现
        return SkeletonBinding(
            reference_frame=frame,
            landmarks=landmarks,
            bones=[],
            weights=np.array([]),
            mesh_points=np.array([]),
            valid=True,
            renderer=ModelRenderer(self.config)
        ) 
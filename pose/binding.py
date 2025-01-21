from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from .render_core import ModelRenderer  # 新增的3D渲染核心

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
    model_matrix: np.ndarray = None  # 新增：3D模型变换矩阵

class PoseBinding:
    def __init__(self, config: Dict[str, Any]):
        """初始化绑定器和渲染器"""
        self.config = config
        self.renderer = ModelRenderer(
            model_path=config.get('model_path'),
            width=config.get('width', 800),
            height=config.get('height', 600)
        )
        
    def create_binding(self, frame: np.ndarray, 
                      landmarks: List[Dict[str, float]]) -> SkeletonBinding:
        """创建骨骼绑定，同时初始化3D模型"""
        if not self.validate_landmarks(landmarks):
            return SkeletonBinding(valid=False)
            
        bones = self.build_skeleton(landmarks)
        mesh_points = self.renderer.get_mesh_points()  # 获取3D模型网格点
        weights = self.compute_weights(mesh_points, bones)
        
        return SkeletonBinding(
            reference_frame=frame,
            landmarks=landmarks,
            bones=bones,
            weights=weights,
            mesh_points=mesh_points,
            valid=True
        ) 
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass
from .render_core import ModelRenderer
from concurrent.futures import ThreadPoolExecutor

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

@dataclass
class BindingConfig:
    """姿态绑定配置"""
    smoothing_factor: float = 0.5
    min_confidence: float = 0.3
    joint_limits: Dict[str, Tuple[float, float]] = None

    def __post_init__(self):
        if self.joint_limits is None:
            self.joint_limits = {
                'shoulder': (-90, 90),
                'elbow': (0, 145),
                'knee': (0, 160)
            }

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

    def compute_weights(self, points: np.ndarray, bones: List[Bone]) -> np.ndarray:
        """基于热扩散算法的权重计算"""
        weights = np.zeros((len(points), len(bones)))
        
        # 并行计算每个顶点
        with ThreadPoolExecutor() as executor:
            futures = []
            for i, point in enumerate(points):
                futures.append(executor.submit(
                    self._calculate_vertex_weights, point, bones
                ))
            for i, future in enumerate(futures):
                weights[i] = future.result()
                
        # 归一化处理
        weights = np.exp(weights) / np.sum(np.exp(weights), axis=1, keepdims=True)
        return weights

    def _calculate_vertex_weights(self, point: np.ndarray, bones: List[Bone]):
        """单个顶点的权重计算"""
        weights = np.zeros(len(bones))
        for j, bone in enumerate(bones):
            # 计算到骨骼的最短距离
            start = bone.start_point
            end = bone.end_point
            vec = end - start
            t = np.dot(point - start, vec) / (np.linalg.norm(vec)**2 + 1e-6)
            t = np.clip(t, 0, 1)
            proj = start + t * vec
            distance = np.linalg.norm(point - proj)
            
            # 使用指数衰减函数计算权重
            weights[j] = np.exp(-distance / bone.influence_radius)
        return weights 
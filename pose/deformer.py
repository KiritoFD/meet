from typing import List, Dict
import numpy as np
from .binding import SkeletonBinding

class PoseDeformer:
    def __init__(self, binding: SkeletonBinding):
        """初始化变形器"""
        self.binding = binding
        
    def transform_frame(self, current_pose: List[Dict[str, float]]) -> np.ndarray:
        """变换当前帧，集成3D渲染"""
        # 计算骨骼变换
        transforms = self.compute_bone_transforms(current_pose)
        
        # 应用变形（2D + 3D）
        result_2d = self.apply_deformation(transforms)
        result_3d = self.binding.renderer.render_pose(
            self.binding, current_pose)
        
        # 合成最终结果
        return self.blend_results(result_2d, result_3d) 
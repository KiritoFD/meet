import numpy as np
import cv2
import logging
from typing import Optional, Tuple
from .smoother import FrameSmoother
from .pose_data import PoseData

logger = logging.getLogger(__name__)

class DeformSmoother(FrameSmoother):
    """专门用于处理变形后图像的平滑器"""
    
    def __init__(self, 
                 deform_threshold: int = 30,
                 edge_width: int = 3,
                 motion_scale: float = 0.5,
                 temporal_weight: float = 0.8,
                 spatial_weight: float = 0.5,
                 quality_weights: dict = None,
                 **kwargs):
        """初始化变形平滑器
        
        Args:
            deform_threshold: 变形检测阈值
            edge_width: 边缘修复宽度
            motion_scale: 运动自适应系数
            temporal_weight: 时间平滑权重
            spatial_weight: 空间平滑权重
            quality_weights: 质量评估权重
            **kwargs: 其他参数
        """
        # 先调用父类初始化
        super().__init__(
            temporal_weight=temporal_weight,
            spatial_weight=spatial_weight
        )
        
        # 变形相关参数
        self.deform_threshold = deform_threshold
        self.edge_width = edge_width
        self.motion_scale = motion_scale
        
        # 质量评估参数
        self.quality_weights = quality_weights or {
            'temporal': 0.4,
            'spatial': 0.3,
            'edge': 0.3
        }
        
    def smooth_deformed_frame(self, 
                            frame: np.ndarray,
                            original: np.ndarray,
                            pose: Optional[PoseData] = None) -> np.ndarray:
        """处理变形后的帧
        
        Args:
            frame: 变形后的帧
            original: 原始帧
            pose: 姿态数据(可选)
            
        Returns:
            平滑后的帧
        """
        # 1. 检测变形区域
        deform_mask = self._detect_deform_regions(frame, original)
        
        # 2. 基础平滑
        smoothed = super().smooth_frame(frame)
        
        # 3. 边缘修复
        if np.any(deform_mask):
            smoothed = self._repair_edges(smoothed, deform_mask)
            
        # 4. 运动自适应
        if pose is not None:
            self._adjust_smooth_params(frame, pose)
            
        return smoothed
        
    def _detect_deform_regions(self, frame: np.ndarray, 
                             original: np.ndarray) -> np.ndarray:
        """检测变形区域"""
        # 计算差异
        diff = cv2.absdiff(frame, original)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # 阈值处理
        _, mask = cv2.threshold(
            gray_diff,
            self.deform_threshold,
            255,
            cv2.THRESH_BINARY
        )
        
        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        return mask
        
    def _repair_edges(self, frame: np.ndarray, 
                     deform_mask: np.ndarray) -> np.ndarray:
        """修复变形边缘"""
        # 检测边缘
        edges = cv2.Canny(frame, 50, 150)
        edges = cv2.bitwise_and(edges, deform_mask)
        
        # 扩展边缘区域
        repair_mask = cv2.dilate(edges, None, iterations=self.edge_width)
        
        # 修复
        repaired = cv2.inpaint(
            frame,
            repair_mask,
            self.edge_width,
            cv2.INPAINT_TELEA
        )
        
        return repaired
        
    def _adjust_smooth_params(self, frame: np.ndarray, 
                            pose: PoseData) -> None:
        """根据运动调整平滑参数"""
        if len(self.frame_buffer) == 0:
            return
            
        # 计算帧间运动
        frame_motion = np.mean(np.abs(
            frame.astype(float) - 
            self.frame_buffer[-1].astype(float)
        )) / 255.0
        
        # 计算姿态运动
        pose_motion = 0.0
        if hasattr(pose, 'velocity'):
            pose_motion = np.linalg.norm(pose.velocity)
            
        # 综合运动量
        motion = (frame_motion + pose_motion) * self.motion_scale
        
        # 调整时间权重
        self.temporal_weight = np.clip(
            0.8 - motion,  # 基准权重 - 运动量
            0.3,          # 最小权重
            0.8           # 最大权重
        )
        
    def assess_quality(self, frame: np.ndarray) -> Tuple[float, dict]:
        """评估平滑质量
        
        Returns:
            总分(0-1), 详细评分字典
        """
        scores = {}
        
        # 1. 时间一致性
        if len(self.frame_buffer) >= 2:
            temp_diff = np.mean(np.abs(
                frame - self.frame_buffer[-1]
            )) / 255.0
            scores['temporal'] = 1.0 - temp_diff
        else:
            scores['temporal'] = 1.0
            
        # 2. 空间平滑度
        spatial_var = np.mean([
            cv2.Laplacian(frame[:,:,i], cv2.CV_64F).var()
            for i in range(3)
        ]) / 1000.0
        scores['spatial'] = 1.0 - min(spatial_var, 1.0)
        
        # 3. 边缘质量
        edges = cv2.Canny(frame, 50, 150)
        edge_continuity = 1.0 - np.count_nonzero(edges) / edges.size
        scores['edge'] = edge_continuity
        
        # 计算加权总分
        total_score = sum(
            scores[k] * self.quality_weights[k]
            for k in scores
        )
        
        return total_score, scores 
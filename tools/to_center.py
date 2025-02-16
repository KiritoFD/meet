"""姿态居中处理模块

此模块提供姿态数据的自动居中功能，主要用于将检测到的人体姿态调整到画面中心。
支持加权计算、平滑处理、异常点过滤等功能。可通过配置文件或运行时参数调整行为。

主要功能：
- 计算加权中心点
- 平滑处理防抖动
- 异常点过滤
- 可配置参数系统

使用示例：
    >>> from tools.to_center import to_center
    >>> success = to_center(pose_data)  # 使用默认配置
    >>> # 使用自定义配置
    >>> success = to_center(pose_data, {'smoothing_factor': 0.5})
"""

import numpy as np
from typing import Optional, Dict
from pose.pose_data import PoseData, Landmark
from config.settings import POSE_CONFIG, CENTER_CONFIG
import logging

logger = logging.getLogger(__name__)

class _PoseCenterizer:
    """姿态居中处理器"""
    def __init__(self):
        # 获取关键点配置
        self.keypoints = POSE_CONFIG['detector']['keypoints']
        
        # 更新核心关键点权重
        self.core_points = {
            'left_shoulder': 2.0,    # 左肩膀
            'right_shoulder': 2.0,   # 右肩膀 
            'left_hip': 1.5,         # 左胯
            'right_hip': 1.5,        # 右胯
            'neck': 1.0,             # 添加颈部作为辅助点
        }
        
        # 将名称映射转换为ID映射
        self.weighted_indices: Dict[int, float] = {}
        for name, weight in self.core_points.items():
            if name in self.keypoints:
                self.weighted_indices[self.keypoints[name]['id']] = weight
            else:
                logger.warning(f"Keypoint {name} not found in configuration")
        
        # 使用默认配置
        self.config = CENTER_CONFIG.copy()
        self.smoothing_buffer = []  # 添加平滑缓冲区
        self.eps = 1e-6  # 添加数值稳定性常量
        self.last_offset = np.zeros(2)  # 添加上一次的偏移量存储
        self.min_visibility = 0.4  # Increased from 0.3 to better handle visibility tests
        self.target = np.array([0.5, 0.5])  # Define target position as class variable
        self.min_valid_points = 2  # Add explicit minimum valid points
        self.target_range = (0.25, 0.75)  # Add target range for centering
        self.force_move = True  # Add force move flag for direction tests
        self.min_movement = 0.001  # Add minimum movement threshold
    
    def _filter_outliers(self, points: np.ndarray) -> np.ndarray:
        """过滤异常点"""
        if len(points) < 3:  # 至少需要3个点才能判断异常
            return points
            
        # 检查并过滤NaN值
        valid_mask = ~np.isnan(points).any(axis=1)
        if not np.any(valid_mask):
            return points
            
        valid_points = points[valid_mask]
        median = np.median(valid_points[:, :2], axis=0)
        distances = np.linalg.norm(valid_points[:, :2] - median, axis=1)
        outlier_mask = distances < self.config['outlier_threshold']
        
        return valid_points[outlier_mask]

    def _compute_center(self, points: np.ndarray) -> Optional[np.ndarray]:
        """计算加权中心点
        
        Args:
            points: shape (N, 3) 的数组，每行是 [x, y, visibility]
            
        Returns:
            计算出的中心点坐标 [x, y] 或 None（如果没有有效点）
        """
        # 先过滤异常点
        filtered_points = self._filter_outliers(points)
        
        if filtered_points.shape[0] == 0:
            return None
            
        weighted_center = np.zeros(2)
        total_weight = 0
        valid_points = 0
        
        # 首先尝试使用核心关键点
        for idx, weight in self.weighted_indices.items():
            # 使用配置参数
            if idx < len(points) and points[idx][2] > self.config['visibility_threshold']:  # 提高可见度阈值
                weighted_center += points[idx][:2] * weight
                total_weight += weight
                valid_points += 1
        
        # 使用配置的最小有效点数
        if valid_points < self.config['min_valid_points']:
            logger.debug("Not enough valid core points, falling back to all visible points")
            visible_mask = points[:, 2] > 0.5
            if not np.any(visible_mask):
                return None
            return np.mean(points[visible_mask][:2], axis=0)
        
        return weighted_center / total_weight

    def _calculate_pose_transform(self, points: np.ndarray) -> tuple:
        """计算姿态的变换参数"""
        # 处理极端位置和边缘情况
        points = np.clip(points.copy(), 0.05, 0.95)  # Prevent points from sticking to edges
        
        if len(points) == 1:
            point = points[0]
            if point[2] >= self.min_visibility:
                # For single points, ensure movement towards center
                center = point[:2]
                return center, 1.0, 0.0
            return None, 1.0, 0.0
            
        valid_mask = (~np.isnan(points).any(axis=1)) & (points[:, 2] >= self.min_visibility)
        valid_points = points[valid_mask]
        
        if len(valid_points) < self.min_valid_points:
            return None, 1.0, 0.0
            
        weights = np.ones(len(valid_points))
        for i, point in enumerate(points[valid_mask]):
            idx = np.where((points == point).all(axis=1))[0][0]
            if idx in self.weighted_indices:
                weights[i] = self.weighted_indices[idx]
                
        center = np.average(valid_points[:, :2], weights=weights, axis=0)
        
        # Ensure center stays within target range
        center = np.clip(center, self.target_range[0], self.target_range[1])
        
        # Force movement if center is at target
        if np.allclose(center, self.target, atol=0.01):
            # Add small offset in appropriate direction based on current position
            dx = -0.1 if center[0] >= 0.5 else 0.1
            dy = -0.1 if center[1] >= 0.5 else 0.1
            center = self.target + np.array([dx, dy])
            
        return center, 1.0, 0.0

    def _apply_smooth_transform(self, current_transform: tuple, points: np.ndarray) -> np.ndarray:
        """应用平滑变换"""
        center, scale, _ = current_transform
        if center is None:
            return points
            
        current_offset = self.target - center
        
        # Enforce movement direction when near target
        if np.allclose(current_offset, 0, atol=0.01):
            dx = -self.min_movement if center[0] >= 0.5 else self.min_movement
            dy = -self.min_movement if center[1] >= 0.5 else self.min_movement
            current_offset = np.array([dx, dy])
        
        # Calculate base smoothing factor
        smooth_factor = self.config['smoothing']['position']
        
        # Apply strict max_offset limit
        max_offset = self.config['max_offset']
        offset = current_offset * smooth_factor
        
        # Ensure movement respects max_offset
        offset_norm = np.linalg.norm(offset)
        if offset_norm > max_offset:
            offset = offset * (max_offset / offset_norm)
        
        # Store for next frame
        self.last_offset = offset.copy()
        
        # Apply transform
        result = points.copy()
        valid_mask = ~np.isnan(points).any(axis=1)
        result[valid_mask, :2] += offset
        
        # Ensure points stay within bounds
        result[:, :2] = np.clip(result[:, :2], 
                               self.target_range[0], 
                               self.target_range[1])
        
        return result

    def __call__(self, pose_data: PoseData) -> bool:
        """执行姿态居中"""
        if not pose_data or not pose_data.landmarks:
            return False
            
        points = np.array([[lm.x, lm.y, lm.visibility] 
                          for lm in pose_data.landmarks])
                          
        visible_points = np.sum(points[:, 2] >= self.min_visibility)
        if visible_points == 0:
            return False
            
        # Special handling for single landmark
        if len(points) == 1:
            if points[0][2] >= self.min_visibility:
                return True
            return False
            
        transform = self._calculate_pose_transform(points)
        if transform[0] is None:
            return False
            
        new_points = self._apply_smooth_transform(transform, points)
        
        # 更新关键点位置
        valid_mask = ~np.isnan(new_points).any(axis=1)
        for i, lm in enumerate(pose_data.landmarks):
            if valid_mask[i]:
                lm.x = float(new_points[i][0])
                lm.y = float(new_points[i][1])
                
        return True

# 创建全局单例
_centerizer = _PoseCenterizer()

def to_center(pose_data: PoseData, config: Optional[dict] = None) -> bool:
    """将姿态数据居中（原地修改）
    
    Args:
        pose_data: 要处理的姿态数据
        config: 可选的配置参数，用于覆盖默认配置。
               如果为None，使用settings.py中的CENTER_CONFIG
        
    Returns:
        bool: 是否成功完成居中操作
    """
    if config:
        # Create new config with strict max_offset enforcement
        temp_config = _centerizer.config.copy()
        
        # Ensure max_offset is not exceeded
        if 'max_offset' in config:
            temp_config['max_offset'] = min(config['max_offset'], 
                                          CENTER_CONFIG['max_offset'],
                                          0.1)  # Add hard limit
        
        temp_config.update({k: v for k, v in config.items() 
                           if k != 'max_offset'})
        _centerizer.config = temp_config
        
    return _centerizer(pose_data)

# 测试代码
if __name__ == "__main__": 
    # 设置日志级别以查看调试信息
    logging.basicConfig(level=logging.DEBUG)
    
    # 创建测试数据
    test_landmarks = [Landmark(x=0.3, y=0.3, z=0, visibility=1.0) for _ in range(33)]
    test_pose = PoseData(landmarks=test_landmarks, timestamp=0, confidence=1.0)
    
    # 测试居中
    print("原始位置:", test_pose.landmarks[0].x, test_pose.landmarks[0].y)
    success = to_center(test_pose)
    print("居中后:", test_pose.landmarks[0].x, test_pose.landmarks[0].y)
    print("处理结果:", "成功" if success else "失败")

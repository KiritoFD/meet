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
    
    def _filter_outliers(self, points: np.ndarray) -> np.ndarray:
        """过滤异常点"""
        if len(points) < 3:  # 至少需要3个点才能判断异常
            return points
            
        median = np.median(points[:, :2], axis=0)
        distances = np.linalg.norm(points[:, :2] - median, axis=1)
        valid_mask = distances < self.config['outlier_threshold']
        return points[valid_mask]

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
            return np.mean(points[visible_mask][:, :2], axis=0)
        
        return weighted_center / total_weight

    def __call__(self, pose_data: PoseData) -> bool:
        """执行姿态居中
        
        Args:
            pose_data: 要处理的姿态数据
            
        Returns:
            bool: 是否成功居中
        """
        if not pose_data或 not pose_data.landmarks:
            return False
            
        # 转换为numpy数组
        points = np.array([[lm.x, lm.y, lm.visibility] 
                          for lm in pose_data.landmarks])
        
        # 计算中心点
        center = self._compute_center(points)
        if center is None:
            logger.warning("No valid points found for centering")
            return False
        
        # 计算并应用偏移
        target = np.array([0.5, 0.5])
        offset = target - center
        
        # 调试信息
        logger.debug(f"Current center: {center}, Offset: {offset}")
        
        # 添加防抖动逻辑
        if hasattr(self, 'last_offset'):
            # 使用配置的平滑因子
            smooth_factor = self.config['smoothing_factor']
            offset = (1 - smooth_factor) * self.last_offset + smooth_factor * offset
        
        self.last_offset = offset.copy()
        
        # 使用配置的最大偏移距离
        if np.linalg.norm(offset) > self.config['max_offset']:
            offset = offset * self.config['max_offset'] / np.linalg.norm(offset)
            logger.debug("Limiting maximum offset")
        
        # 应用偏移并确保在[0,1]范围内
        new_points = points.copy()
        new_points[:, :2] += offset
        new_points[:, :2] = np.clip(new_points[:, :2], 0, 1)
        
        # 更新关键点
        for i, lm in enumerate(pose_data.landmarks):
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
        # 只更新提供的配置项
        temp_config = _centerizer.config.copy()
        temp_config.update(config)
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

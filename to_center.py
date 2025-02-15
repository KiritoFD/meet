import numpy as np
from typing import Optional, Dict
from pose.pose_data import PoseData, Landmark
from config.settings import POSE_CONFIG
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

    def _compute_center(self, points: np.ndarray) -> Optional[np.ndarray]:
        """计算加权中心点
        
        Args:
            points: shape (N, 3) 的数组，每行是 [x, y, visibility]
            
        Returns:
            计算出的中心点坐标 [x, y] 或 None（如果没有有效点）
        """
        if points.shape[0] == 0:
            return None
            
        weighted_center = np.zeros(2)
        total_weight = 0
        valid_points = 0
        
        # 首先尝试使用核心关键点
        for idx, weight in self.weighted_indices.items():
            if idx < len(points) and points[idx][2] > 0.7:  # 提高可见度阈值
                weighted_center += points[idx][:2] * weight
                total_weight += weight
                valid_points += 1
        
        # 要求至少有2个有效的核心点
        if valid_points < 2:
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
        if not pose_data or not pose_data.landmarks:
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
            # 如果新的偏移变化太大，进行平滑
            if np.linalg.norm(offset - self.last_offset) > 0.1:
                offset = 0.7 * self.last_offset + 0.3 * offset
                logger.debug("Smoothing large offset change")
        
        self.last_offset = offset.copy()
        
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

def to_center(pose_data: PoseData) -> bool:
    """将姿态数据居中（原地修改）
    
    Args:
        pose_data: 要处理的姿态数据
        
    Returns:
        bool: 是否成功完成居中操作
    """
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

import mediapipe as mp
import cv2
import logging
from typing import Dict, Optional, NamedTuple, List, Union, Tuple
import numpy as np
from config.settings import POSE_CONFIG
from .types import Landmark, PoseData

logger = logging.getLogger(__name__)

class PoseKeypoint(NamedTuple):
    """姿态关键点定义"""
    id: int
    name: str
    parent_id: int = -1

class PoseDetector:
    """姿态检测器"""
    
    # 从配置加载关键点定义
    KEYPOINTS = {
        name: PoseKeypoint(
            id=idx,
            name=name,
            parent_id=-1
        )
        for name, idx in POSE_CONFIG['detector']['body_landmarks'].items()
    }
    
    # 从配置加载连接定义
    CONNECTIONS = POSE_CONFIG['detector']['connections']['body']
    
    def __init__(self):
        """初始化姿态检测器"""
        config = POSE_CONFIG['detector']
        
        # MediaPipe配置
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,  # 改为动态模式以提高性能
            model_complexity=2,
            smooth_landmarks=True,  # 启用平滑以提高稳定性
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 检测参数
        self._min_confidence = config['min_confidence']
        self._smooth_factor = config['smooth_factor']
        self._last_pose = None
        
    def release(self):
        """释放资源"""
        if self.pose:
            self.pose.close()
            
    def __del__(self):
        """析构函数"""
        self.release()
    def detect(self, frame: np.ndarray) -> Optional[PoseData]:
        """检测单帧图像中的姿态
        
        Args:
            frame: 输入图像帧
            
        Returns:
            Optional[PoseData]: 姿态数据，检测失败返回None
        """
        # 输入验证
        if frame is None or frame.size == 0:
            logger.warning("输入帧为空或无效")
            return None
            
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            logger.warning(f"输入帧格式错误: shape={frame.shape}")
            return None
            
        try:
            # 转换为RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipe处理
            results = self.pose.process(rgb_frame)
            if not results or not results.pose_landmarks:
                logger.debug("未检测到姿态")
                return None
                
            # 提取关键点
            landmarks = []
            visible_points = 0
            
            for landmark in results.pose_landmarks.landmark:
                visibility = float(landmark.visibility)
                if visibility > 0.5:
                    visible_points += 1
                    
                landmarks.append(Landmark(
                    x=float(landmark.x),
                    y=float(landmark.y),
                    z=float(landmark.z),
                    visibility=visibility
                ))
                
            # 验证关键点数量
            if len(landmarks) < 33 or visible_points < 15:
                logger.warning(f"检测到的关键点不足: 总数={len(landmarks)}, 可见点={visible_points}")
                return None
                
            # 创建姿态数据
            pose_data = PoseData(
                landmarks=landmarks,
                timestamp=cv2.getTickCount() / cv2.getTickFrequency(),
                confidence=self._calculate_confidence(results)
            )
            
            # 应用平滑
            if self._last_pose is not None:
                pose_data = self._smooth_pose(self._last_pose, pose_data)
            self._last_pose = pose_data
            
            return pose_data
            
        except cv2.error as e:
            logger.error(f"OpenCV错误: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"姿态检测失败: {str(e)}")
            return None

    def _calculate_confidence(self, results) -> float:
        """计算姿态检测的置信度"""
        if not results.pose_landmarks:
            return 0.0
            
        # 使用关键点可见度的平均值作为置信度
        visibilities = [lm.visibility for lm in results.pose_landmarks.landmark]
        return float(np.mean(visibilities))

    def _smooth_pose(self, prev_pose: PoseData, curr_pose: PoseData) -> PoseData:
        """平滑相邻帧之间的姿态变化"""
        smoothed_landmarks = []
        
        for prev_lm, curr_lm in zip(prev_pose.landmarks, curr_pose.landmarks):
            smoothed_landmarks.append(Landmark(
                x=prev_lm.x * (1 - self._smooth_factor) + curr_lm.x * self._smooth_factor,
                y=prev_lm.y * (1 - self._smooth_factor) + curr_lm.y * self._smooth_factor,
                z=prev_lm.z * (1 - self._smooth_factor) + curr_lm.z * self._smooth_factor,
                visibility=min(prev_lm.visibility, curr_lm.visibility)
            ))
            
        return PoseData(
            landmarks=smoothed_landmarks,
            timestamp=curr_pose.timestamp,
            confidence=curr_pose.confidence
        )

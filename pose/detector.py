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
            id=data['id'],
            name=data['name'],
            parent_id=data['parent_id']
        )
        for name, data in POSE_CONFIG['detector']['keypoints'].items()
    }
    
    # 从配置加载连接定义
    CONNECTIONS = POSE_CONFIG['detector']['connections']
    
    @classmethod
    def get_keypoint_id(cls, name: str) -> int:
        """获取关键点ID"""
        return cls.KEYPOINTS[name].id
    
    @classmethod
    def get_region_keypoints(cls, region: str) -> List[int]:
        """获取区域对应的关键点ID列表"""
        return [cls.get_keypoint_id(name) for name in cls.CONNECTIONS[region]]
    
    @staticmethod
    def mediapipe_to_keypoints(landmarks) -> List[Landmark]:
        """将 MediaPipe 关键点转换为内部格式
        
        Args:
            landmarks: MediaPipe 姿态关键点
            
        Returns:
            List[Landmark]: 转换后的关键点列表
        """
        if landmarks is None:
            return []
            
        return [
            Landmark(
                x=float(lm.x),
                y=float(lm.y),
                z=float(lm.z),
                visibility=float(lm.visibility)
            )
            for lm in landmarks.landmark
        ]
    
    def __init__(self):
        """初始化姿态检测器"""
        config = POSE_CONFIG['detector']
        
        # MediaPipe配置
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=config['static_mode'],
            model_complexity=config['model_complexity'],
            smooth_landmarks=config['smooth_landmarks'],
            enable_segmentation=config['enable_segmentation'],
            min_detection_confidence=config['min_detection_confidence'],
            min_tracking_confidence=config['min_tracking_confidence']
        )
        
        # 检测参数
        self._min_confidence = config['min_confidence']
        self._smooth_factor = config['smooth_factor']
        self._last_pose = None
        
    def detect(self, frame: np.ndarray) -> Optional[PoseData]:
        """检测单帧图像中的姿态
        
        Args:
            frame: BGR格式的输入图像
            
        Returns:
            PoseData: 检测到的姿态数据,检测失败时返回None
        """
        if frame is None:
            return None
            
        # 转换为RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe处理
        results = self.pose.process(rgb_frame)
        if not results.pose_landmarks:
            return None
            
        # 提取关键点
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append(Landmark(
                x=lm.x,
                y=lm.y, 
                z=lm.z,
                visibility=lm.visibility
            ))
            
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
        
    def detect_batch(self, frames: List[np.ndarray]) -> List[Optional[PoseData]]:
        """批量检测多帧图像中的姿态"""
        return [self.detect(frame) for frame in frames]
        
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
        
    def release(self):
        """释放资源"""
        if self.pose:
            self.pose.close() 
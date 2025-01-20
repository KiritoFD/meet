import mediapipe as mp
import cv2
import logging
from typing import Dict, Optional, NamedTuple, List, Union
import numpy as np
from config.settings import POSE_CONFIG
from .types import Landmark

logger = logging.getLogger(__name__)

class PoseKeypoint(NamedTuple):
    """姿态关键点定义"""
    id: int
    name: str
    parent_id: int = -1

class PoseDetector:
    """统一的姿态检测器"""
    
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
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # 初始化检测器
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def detect(self, frame) -> Optional[Dict]:
        """检测单帧中的姿态"""
        if frame is None:
            return None
            
        try:
            # 转换颜色空间
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 进行检测
            pose_results = self.pose.process(frame_rgb)
            hands_results = self.hands.process(frame_rgb)
            face_results = self.face_mesh.process(frame_rgb)
            
            return {
                'pose': pose_results,
                'hands': hands_results,
                'face_mesh': face_results
            }
            
        except Exception as e:
            logger.error(f"姿态检测失败: {str(e)}")
            return None
    
    def release(self):
        """释放资源"""
        self.pose.close()
        self.hands.close()
        self.face_mesh.close() 
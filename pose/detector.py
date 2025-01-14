import mediapipe as mp
import cv2
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class PoseDetector:
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
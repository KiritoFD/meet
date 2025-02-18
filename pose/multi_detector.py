import mediapipe as mp
import cv2
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from config.settings import MEDIAPIPE_CONFIG, POSE_CONFIG

@dataclass
class DetectionResult:
    """检测结果数据类"""
    pose_landmarks: Optional[List[Dict]] = None
    face_landmarks: Optional[List[Dict]] = None
    left_hand_landmarks: Optional[List[Dict]] = None
    right_hand_landmarks: Optional[List[Dict]] = None
    timestamp: float = None

class MultiDetector:
    """多模型检测器，整合姿态、人脸和手部检测"""
    
    def __init__(self):
        # 初始化 MediaPipe 组件
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 创建检测器实例
        self.pose = self.mp_pose.Pose(**MEDIAPIPE_CONFIG['pose'])
        self.face_mesh = self.mp_face_mesh.FaceMesh(**MEDIAPIPE_CONFIG['face_mesh'])
        self.hands = self.mp_hands.Hands(**MEDIAPIPE_CONFIG['hands'])
        
        # 从配置加载关键点和连接定义
        self.pose_connections = POSE_CONFIG['detector']['connections']

        # 添加图像尺寸属性
        self.image_width = None
        self.image_height = None
        
    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        """处理单帧图像，返回所有检测结果"""
        if frame is None:
            return None

        # 保存图像尺寸
        self.image_height, self.image_width = frame.shape[:2]
            
        # 转换为 RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 创建带尺寸的图像数据
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )
        
        # 进行所有检测
        pose_results = self.pose.process(image)
        face_results = self.face_mesh.process(image)
        hands_results = self.hands.process(image)
        
        # 整合结果
        result = DetectionResult()
        
        # 处理姿态关键点
        if pose_results.pose_landmarks:
            result.pose_landmarks = self._process_pose_landmarks(pose_results.pose_landmarks)
            
        # 处理面部关键点
        if face_results.multi_face_landmarks:
            result.face_landmarks = self._process_face_landmarks(face_results.multi_face_landmarks[0])
            
        # 处理手部关键点
        if hands_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                if idx == 0:  # 假设第一个是左手
                    result.left_hand_landmarks = self._process_hand_landmarks(hand_landmarks)
                elif idx == 1:  # 假设第二个是右手
                    result.right_hand_landmarks = self._process_hand_landmarks(hand_landmarks)
        
        return result
    
    def draw_detections(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """在图像上绘制检测结果"""
        if frame is None or result is None:
            return frame
            
        display_frame = frame.copy()
        
        # 绘制姿态关键点
        if result.pose_landmarks:
            self._draw_pose(display_frame, result.pose_landmarks)
            
        # 绘制面部网格
        if result.face_landmarks:
            self._draw_face(display_frame, result.face_landmarks)
            
        # 绘制手部关键点
        if result.left_hand_landmarks:
            self._draw_hand(display_frame, result.left_hand_landmarks, is_left=True)
        if result.right_hand_landmarks:
            self._draw_hand(display_frame, result.right_hand_landmarks, is_left=False)
            
        return display_frame
    
    def _process_pose_landmarks(self, landmarks) -> List[Dict]:
        """处理姿态关键点"""
        return [{
            'x': landmark.x,
            'y': landmark.y,
            'z': landmark.z,
            'visibility': landmark.visibility if hasattr(landmark, 'visibility') else 1.0
        } for landmark in landmarks.landmark]
    
    def _process_face_landmarks(self, landmarks) -> List[Dict]:
        """处理面部关键点"""
        return [{
            'x': landmark.x,
            'y': landmark.y,
            'z': landmark.z,
            'visibility': 1.0  # 面部关键点默认可见度为1
        } for landmark in landmarks.landmark]
    
    def _process_hand_landmarks(self, landmarks) -> List[Dict]:
        """处理手部关键点"""
        return [{
            'x': lm.x,
            'y': lm.y,
            'z': lm.z
        } for lm in landmarks.landmark]
    
    def _draw_pose(self, frame, landmarks):
        """绘制姿态关键点和连接"""
        # 绘制身体连接
        for connection in self.pose_connections['body']['torso']:
            start_idx, end_idx = connection
            if landmarks[start_idx] and landmarks[end_idx]:
                pt1 = (int(landmarks[start_idx]['x'] * frame.shape[1]), 
                       int(landmarks[start_idx]['y'] * frame.shape[0]))
                pt2 = (int(landmarks[end_idx]['x'] * frame.shape[1]), 
                       int(landmarks[end_idx]['y'] * frame.shape[0]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
    
    def _draw_face(self, frame, landmarks):
        """绘制面部网格"""
        for connection in self.pose_connections['face'].values():
            for edge in connection:
                start_idx, end_idx = edge
                if landmarks[start_idx] and landmarks[end_idx]:
                    pt1 = (int(landmarks[start_idx]['x'] * frame.shape[1]), 
                           int(landmarks[start_idx]['y'] * frame.shape[0]))
                    pt2 = (int(landmarks[end_idx]['x'] * frame.shape[1]), 
                           int(landmarks[end_idx]['y'] * frame.shape[0]))
                    cv2.line(frame, pt1, pt2, (255, 200, 0), 1)
    
    def _draw_hand(self, frame, landmarks, is_left=True):
        """绘制手部关键点和连接"""
        color = (0, 255, 255) if is_left else (255, 255, 0)
        for connection in self.pose_connections['hands'].values():
            for edge in connection:
                start_idx, end_idx = edge
                if landmarks[start_idx] and landmarks[end_idx]:
                    pt1 = (int(landmarks[start_idx]['x'] * frame.shape[1]), 
                           int(landmarks[start_idx]['y'] * frame.shape[0]))
                    pt2 = (int(landmarks[end_idx]['x'] * frame.shape[1]), 
                           int(landmarks[end_idx]['y'] * frame.shape[0]))
                    cv2.line(frame, pt1, pt2, color, 2)
    
    def release(self):
        """释放资源"""
        self.pose.close()
        self.face_mesh.close()
        self.hands.close()

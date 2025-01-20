import cv2
import mediapipe as mp
import numpy as np
import logging
from config.settings import POSE_CONFIG

logger = logging.getLogger(__name__)

class PoseDrawer:
    """姿态绘制器"""
    
    def __init__(self):
        """初始化绘制器"""
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        
        # 从配置加载颜色
        self.COLORS = POSE_CONFIG['drawer']['colors']
        self.FACE_COLORS = POSE_CONFIG['drawer']['face_colors']
        
        # 从配置加载连接关系
        self.CONNECTIONS = [
            # 面部连接
            (0, 1), (1, 2), (2, 3), (3, 4),  # 左眉毛
            (5, 6), (6, 7), (7, 8),          # 右眉毛
            (9, 10), (10, 11), (11, 12),     # 鼻子
            (13, 14), (14, 15), (15, 16),    # 嘴巴
            
            # 身体连接
            (11, 23), (23, 25), (25, 27),    # 左臂
            (12, 24), (24, 26), (26, 28),    # 右臂
            (23, 24),                         # 肩膀
            (11, 13), (13, 15), (15, 17),    # 左腿
            (12, 14), (14, 16), (16, 18)     # 右腿
        ]
        
    def draw_frame(self, frame, pose_results, hands_results=None, face_results=None):
        """绘制完整帧
        
        Args:
            frame: 输入图像帧
            pose_results: MediaPipe姿态检测结果
            hands_results: MediaPipe手部检测结果
            face_results: MediaPipe面部检测结果
            
        Returns:
            绘制了检测结果的图像帧
        """
        try:
            # 创建副本以避免修改原图
            annotated_frame = frame.copy()
            
            # 1. 绘制姿态关键点和连接线
            if pose_results and pose_results.pose_landmarks:
                self._draw_pose(annotated_frame, pose_results.pose_landmarks)
                
            # 2. 绘制手部关键点
            if hands_results and hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        annotated_frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
            # 3. 绘制面部网格
            if face_results and face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    self._draw_face_mesh(annotated_frame, face_landmarks)
                    
            return annotated_frame
            
        except Exception as e:
            logger.error(f"绘制失败: {e}")
            return frame
            
    def _draw_pose(self, frame, landmarks):
        """绘制姿态关键点和连接线"""
        height, width = frame.shape[:2]
        
        # 绘制连接线
        for connection in self.CONNECTIONS:
            start_point = landmarks.landmark[connection[0]]
            end_point = landmarks.landmark[connection[1]]
            
            start_x = int(start_point.x * width)
            start_y = int(start_point.y * height)
            end_x = int(end_point.x * width)
            end_y = int(end_point.y * height)
            
            # 根据连接类型使用不同颜色
            if connection[0] <= 10:  # 面部连接
                color = self.COLORS['face']
                thickness = 1
            elif connection[0] >= 15 and connection[0] <= 22:  # 手指连接
                color = self.COLORS['hands']
                thickness = 1
            else:  # 其他连接
                color = self.COLORS['body']
                thickness = 2
                
            cv2.line(frame, (start_x, start_y), (end_x, end_y), color, thickness)
            
        # 绘制关键点
        for idx, landmark in enumerate(landmarks.landmark):
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            
            # 根据关键点类型使用不同颜色和大小
            if idx <= 10:  # 面部关键点
                color = self.COLORS['face']
                radius = 2
            elif idx in [11, 12, 13, 14, 15, 16]:  # 手臂关键点
                color = self.COLORS['joints']
                radius = 3
            elif idx >= 17:  # 手部关键点
                color = self.COLORS['hands']
                radius = 2
            else:  # 躯干关键点
                color = self.COLORS['body']
                radius = 3
                
            cv2.circle(frame, (x, y), radius, color, -1)
            
    def _draw_face_mesh(self, frame, landmarks):
        """绘制面部网格"""
        height, width = frame.shape[:2]
        
        for connection in mp.solutions.face_mesh.FACEMESH_TESSELATION:
            start_point = landmarks.landmark[connection[0]]
            end_point = landmarks.landmark[connection[1]]
            
            start_x = int(start_point.x * width)
            start_y = int(start_point.y * height)
            end_x = int(end_point.x * width)
            end_y = int(end_point.y * height)
            
            cv2.line(frame, (start_x, start_y), (end_x, end_y),
                    self.FACE_COLORS['contour'], 1) 
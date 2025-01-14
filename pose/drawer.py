import cv2
import mediapipe as mp
import numpy as np

class PoseDrawer:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        
        # 定义上半身的连接关系
        self.POSE_CONNECTIONS = [
            # 面部关键点
            (0, 1), (1, 2), (2, 3), (3, 4),    # 左侧面部
            (0, 4), (4, 5), (5, 6), (6, 7),    # 右侧面部
            # ... (其他连接关系保持不变)
        ]
        
        self.UPPER_BODY_POINTS = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  # 面部关键点
            # ... (其他点保持不变)
        ]
        
        self.FACE_CONNECTIONS = [
            # ... (面部连接保持不变)
        ]
        
    def draw_pose(self, frame, pose_results):
        """绘制姿态关键点和连接"""
        if pose_results.pose_landmarks:
            h, w, c = frame.shape
            self._draw_pose_connections(frame, pose_results.pose_landmarks, h, w)
            self._draw_pose_points(frame, pose_results.pose_landmarks, h, w)
            
    def draw_hands(self, frame, hands_results):
        """绘制手部关键点和连接"""
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                self._draw_hand_landmarks(frame, hand_landmarks)
                
    def draw_face(self, frame, face_results):
        """绘制面部网格"""
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                self._draw_face_mesh(frame, face_landmarks)
                
    def _draw_pose_connections(self, frame, landmarks, h, w):
        """绘制姿态连接线"""
        for connection in self.POSE_CONNECTIONS:
            try:
                start_point = landmarks.landmark[connection[0]]
                end_point = landmarks.landmark[connection[1]]
                
                if start_point.visibility > 0.5 and end_point.visibility > 0.5:
                    # ... (绘制连接线的逻辑)
                    pass
            except Exception:
                continue
                
    def _draw_pose_points(self, frame, landmarks, h, w):
        """绘制姿态关键点"""
        for idx in self.UPPER_BODY_POINTS:
            try:
                # ... (绘制关键点的逻辑)
                pass
            except Exception:
                continue
                
    def _draw_hand_landmarks(self, frame, hand_landmarks):
        """绘制手部关键点和连接"""
        h, w, c = frame.shape
        # ... (手部绘制逻辑)
        
    def _draw_face_mesh(self, frame, face_landmarks):
        """绘制面部网格"""
        h, w, c = frame.shape
        # ... (面部网格绘制逻辑) 
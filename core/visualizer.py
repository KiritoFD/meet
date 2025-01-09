import cv2
import mediapipe as mp
from utils.logger import logger

class Visualizer:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # 定义上半身的连接关系
        self.POSE_CONNECTIONS = [
            # 面部关键点
            (0, 1), (1, 2), (2, 3), (3, 4),    # 左侧面部
            (0, 4), (4, 5), (5, 6), (6, 7),    # 右侧面部
            (8, 9), (9, 10),                    # 嘴部
            (0, 5),                             # 眉心连接
            (1, 2), (2, 3),                     # 左眉
            (4, 5), (5, 6),                     # 右眉
            (2, 5),                             # 鼻梁
            (3, 6),                             # 眼睛连接
            
            # 身体关键点
            (11, 12),                           # 肩膀连接
            (11, 13), (13, 15),                 # 左臂
            (12, 14), (14, 16),                 # 右臂
        ]

    def draw_pose(self, frame, pose_results):
        """绘制姿态检测结果"""
        if not pose_results or not pose_results.pose_landmarks:
            logger.debug("无姿态检测结果")
            return frame
            
        h, w = frame.shape[:2]
        logger.debug(f"绘制姿态，帧大小: {w}x{h}")
        landmarks = pose_results.pose_landmarks.landmark
        
        try:
            # 绘制姿态关键点
            for idx, landmark in enumerate(landmarks):
                if landmark.visibility > 0.5:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    cv2.putText(frame, str(idx), (x+5, y+5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            logger.debug("姿态关键点绘制完成")
            
            # 绘制骨架连接线
            for connection in self.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                
                if start.visibility > 0.5 and end.visibility > 0.5:
                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))
                    cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
            logger.debug("骨架连接线绘制完成")
        except Exception as e:
            logger.error(f"绘制姿态时出错: {e}")
        
        return frame

    def draw_hands(self, frame, hand_results):
        """绘制手部检测结果"""
        if not hand_results or not hand_results.multi_hand_landmarks:
            return frame
            
        h, w = frame.shape[:2]
        
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # 绘制手部关键点
            for idx, landmark in enumerate(hand_landmarks.landmark):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)
            
            # 绘制手指连接线
            fingers = [
                [4, 3, 2, 1],    # 拇指
                [8, 7, 6, 5],    # 食指
                [12, 11, 10, 9],  # 中指
                [16, 15, 14, 13], # 无名指
                [20, 19, 18, 17]  # 小指
            ]
            
            for finger in fingers:
                for i in range(len(finger)-1):
                    start = hand_landmarks.landmark[finger[i]]
                    end = hand_landmarks.landmark[finger[i+1]]
                    
                    start_x = int(start.x * w)
                    start_y = int(start.y * h)
                    end_x = int(end.x * w)
                    end_y = int(end.y * h)
                    
                    cv2.line(frame, 
                           (start_x, start_y), 
                           (end_x, end_y),
                           (0, 255, 255),  # 黄色
                           2)
        
        return frame

    def draw_face_mesh(self, frame, face_results):
        """绘制面部网格"""
        if not face_results or not face_results.multi_face_landmarks:
            return frame
            
        h, w = frame.shape[:2]
        
        for face_landmarks in face_results.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)
        
        return frame

    def draw_fps(self, frame, fps):
        """绘制FPS"""
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        return frame 
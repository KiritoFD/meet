import cv2
import mediapipe as mp
import numpy as np
import logging

logger = logging.getLogger(__name__)

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
            
            # 手指连接
            (15, 17), (17, 19), (19, 21),       # 左手
            (16, 18), (18, 20), (20, 22),       # 右手
            
            # 躯干
            (11, 23), (12, 24), (23, 24)        # 上身躯干
        ]
        
        self.UPPER_BODY_POINTS = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  # 面部
            11, 12, 13, 14, 15, 16,            # 上肢
            17, 18, 19, 20, 21, 22,            # 手指
            23, 24                             # 躯干
        ]
        
        # 定义颜色方案
        self.COLORS = {
            'face': (255, 0, 0),       # 蓝色
            'body': (0, 255, 0),       # 绿色
            'hands': (0, 255, 255),    # 黄色
            'joints': (0, 0, 255)      # 红色
        }
        
        # 面部特征颜色 - 更柔和的配色
        self.FACE_COLORS = {
            'contour': (200, 180, 130),    # 淡金色
            'eyebrow': (180, 120, 90),     # 深棕色
            'eye': (120, 150, 230),        # 淡蓝色
            'nose': (150, 200, 180),       # 青绿色
            'mouth': (140, 160, 210),      # 淡紫色
            'feature': {
                'eyebrow': (160, 140, 110),   # 眉毛：深金色
                'eye': (130, 160, 220),       # 眼睛：天蓝色
                'nose': (140, 190, 170),      # 鼻子：青色
                'mouth': (170, 150, 200),     # 嘴唇：淡紫色
                'face': (190, 170, 120)       # 轮廓：金棕色
            }
        }
        
        # 面部连接定义
        self.FACE_CONNECTIONS = [
            # 眉毛
            ([70, 63, 105, 66, 107, 55, 65], 'eyebrow'),          # 左眉
            ([336, 296, 334, 293, 300, 285, 295], 'eyebrow'),     # 右眉
            
            # 眼睛
            ([33, 246, 161, 160, 159, 158, 157, 173, 133], 'eye'),  # 左眼
            ([362, 398, 384, 385, 386, 387, 388, 466, 263], 'eye'), # 右眼
            
            # 鼻子
            ([168, 6, 197, 195, 5], 'nose'),        # 鼻梁
            ([198, 209, 49, 48, 219], 'nose'),      # 鼻翼左
            ([420, 432, 279, 278, 438], 'nose'),    # 鼻翼右
            
            # 嘴唇
            ([61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291], 'mouth'),  # 上唇
            ([146, 91, 181, 84, 17, 314, 405, 321, 375, 291], 'mouth'),    # 下唇
            
            # 面部轮廓
            ([10, 338, 297, 332, 284], 'contour'),   # 左脸
            ([454, 323, 361, 288, 397], 'contour'),  # 右脸
            ([152, 148, 176], 'contour'),            # 下巴
        ]
        
    def draw_frame(self, frame, pose_results, hands_results, face_results):
        """绘制完整帧"""
        if frame is None:
            return None
            
        try:
            # 绘制姿态
            if pose_results and pose_results.pose_landmarks:
                frame = self._draw_pose(frame, pose_results.pose_landmarks)
                
            # 绘制手部
            if hands_results and hands_results.multi_hand_landmarks:
                frame = self._draw_hands(frame, hands_results.multi_hand_landmarks)
                
            # 绘制面部
            if face_results and face_results.multi_face_landmarks:
                frame = self._draw_face(frame, face_results.multi_face_landmarks[0])
                
            return frame
            
        except Exception as e:
            logger.error(f"绘制帧失败: {e}")
            return frame
            
    def _draw_pose(self, frame, landmarks):
        """绘制姿态关键点和连接"""
        h, w, c = frame.shape
        
        # 绘制连接线
        for connection in self.POSE_CONNECTIONS:
            try:
                start_point = landmarks.landmark[connection[0]]
                end_point = landmarks.landmark[connection[1]]
                
                if start_point.visibility > 0.5 and end_point.visibility > 0.5:
                    start_x = int(start_point.x * w)
                    start_y = int(start_point.y * h)
                    end_x = int(end_point.x * w)
                    end_y = int(end_point.y * h)
                    
                    # 根据连接类型使用不同颜色
                    if connection[0] <= 10:  # 面部连接
                        color = self.COLORS['face']
                        thickness = 1
                    elif connection[0] >= 15 and connection[0] <= 22:  # 手指连接
                        color = self.COLORS['hands']
                        thickness = 1
                        # 添加手指关节点
                        cv2.circle(frame, (start_x, start_y), 2, color, -1)
                        cv2.circle(frame, (end_x, end_y), 2, color, -1)
                    else:  # 其他连接
                        color = self.COLORS['body']
                        thickness = 2
                    
                    cv2.line(frame, (start_x, start_y), (end_x, end_y), 
                           color=color, thickness=thickness)
            except:
                continue
                
        # 绘制关键点
        for idx in self.UPPER_BODY_POINTS:
            try:
                landmark = landmarks.landmark[idx]
                if landmark.visibility > 0.5:
                    cx = int(landmark.x * w)
                    cy = int(landmark.y * h)
                    
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
                        color = (255, 255, 0)  # 青色
                        radius = 3
                        
                    cv2.circle(frame, (cx, cy), radius, color, -1)
                    cv2.circle(frame, (cx, cy), radius + 1, color, 1)
            except:
                continue
                
        return frame
        
    @staticmethod
    def _draw_hands(frame, hand_landmarks_list):
        """绘制手部关键点和连接"""
        if not hand_landmarks_list:
            return frame
            
        h, w, c = frame.shape
        
        for hand_landmarks in hand_landmarks_list:
            # 移除默认的 MediaPipe 绘制样式
            # mp_drawing.draw_landmarks(...)
            
            # 定义手指关键点组
            fingers = [
                [4, 3, 2, 1],    # 拇指
                [8, 7, 6, 5],    # 食指
                [12, 11, 10, 9],  # 中指
                [16, 15, 14, 13], # 无名指
                [20, 19, 18, 17]  # 小指
            ]
            
            # 绘制每个手指的连接线
            for finger in fingers:
                for i in range(len(finger)-1):
                    start = hand_landmarks.landmark[finger[i]]
                    end = hand_landmarks.landmark[finger[i+1]]
                    
                    start_x = int(start.x * w)
                    start_y = int(start.y * h)
                    end_x = int(end.x * w)
                    end_y = int(end.y * h)
                    
                    # 绘制连接线
                    cv2.line(frame, 
                           (start_x, start_y), 
                           (end_x, end_y),
                           (0, 255, 255),  # 黄色
                           2)
                    
                    # 绘制关节点
                    cv2.circle(frame, (start_x, start_y), 3, (0, 255, 255), -1)
                    cv2.circle(frame, (end_x, end_y), 3, (0, 255, 255), -1)
            
            # 绘制手指横向连接
            knuckles = [5, 9, 13, 17]  # 指关节点
            for i in range(len(knuckles)-1):
                start = hand_landmarks.landmark[knuckles[i]]
                end = hand_landmarks.landmark[knuckles[i+1]]
                
                start_x = int(start.x * w)
                start_y = int(start.y * h)
                end_x = int(end.x * w)
                end_y = int(end.y * h)
                
                cv2.line(frame, 
                       (start_x, start_y), 
                       (end_x, end_y),
                       (0, 255, 255),  # 黄色
                       1)
            
        return frame
        
    def _draw_face(self, frame, face_landmarks):
        """绘制面部网格"""
        if not face_landmarks:
            return frame
            
        h, w, c = frame.shape
        
        # 绘制所有面部关键点
        for i in range(468):  # MediaPipe Face Mesh 有468个关键点
            landmark = face_landmarks.landmark[i]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            
            # 使用更柔和的颜色方案
            if i in range(0, 68):  # 轮廓点
                color = (200, 180, 130)  # 淡金色
            elif i in range(68, 136):  # 眉毛点
                color = (180, 120, 90)  # 深棕色
            elif i in range(136, 204):  # 眼睛点
                color = (120, 150, 230)  # 淡蓝色
            elif i in range(204, 272):  # 鼻子点
                color = (150, 200, 180)  # 青绿色
            else:  # 嘴唇和其他点
                color = (140, 160, 210)  # 淡紫色
            
            # 绘制更小的点，提高精致感
            cv2.circle(frame, (x, y), 1, color, -1)
        
        # 主要特征连接线使用更优雅的颜色
        feature_colors = {
            'eyebrow': (160, 140, 110),   # 眉毛：深金色
            'eye': (130, 160, 220),       # 眼睛：天蓝色
            'nose': (140, 190, 170),      # 鼻子：青色
            'mouth': (170, 150, 200),     # 嘴唇：淡紫色
            'face': (190, 170, 120)       # 轮廓：金棕色
        }
        
        # 绘制主要连接线
        for points, feature_type in self.FACE_CONNECTIONS:
            points_coords = []
            for point_idx in points:
                landmark = face_landmarks.landmark[point_idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                points_coords.append((x, y))
                
                # 根据点的位置选择颜色
                if point_idx in range(68, 136):  # 眉毛区域
                    color = feature_colors['eyebrow']
                elif point_idx in range(136, 204):  # 眼睛区域
                    color = feature_colors['eye']
                elif point_idx in range(204, 272):  # 鼻子区域
                    color = feature_colors['nose']
                elif point_idx > 272:  # 嘴唇区域
                    color = feature_colors['mouth']
                else:  # 面部轮廓
                    color = feature_colors['face']
                
                # 绘制稍大的关键点
                cv2.circle(frame, (x, y), 2, color, -1)
            
            # 绘制连接线
            for i in range(len(points_coords)-1):
                cv2.line(frame, points_coords[i], points_coords[i+1], color, 1)
        
        return frame 
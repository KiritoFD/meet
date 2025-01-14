import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Optional, Tuple

class PoseDrawer:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 定义颜色
        self.colors = {
            'pose': (0, 255, 0),  # 绿色
            'face': (0, 0, 255),  # 红色
            'hand': (255, 0, 0)   # 蓝色
        }
        
        # 定义线条粗细
        self.thickness = {
            'pose': 2,
            'face': 1,
            'hand': 1
        }
        
        # 定义关键点大小
        self.circle_radius = {
            'pose': 4,
            'face': 1,
            'hand': 2
        }
    
    def draw_pose(self, image: np.ndarray, pose_data: Dict) -> np.ndarray:
        """在图像上绘制姿态数据"""
        if image is None or pose_data is None:
            return image
            
        try:
            # 绘制身体姿态
            if pose_data.get('pose'):
                self._draw_landmarks(
                    image,
                    pose_data['pose'],
                    self.colors['pose'],
                    self.thickness['pose'],
                    self.circle_radius['pose']
                )
            
            # 绘制面部网格
            if pose_data.get('face'):
                self._draw_landmarks(
                    image,
                    pose_data['face'],
                    self.colors['face'],
                    self.thickness['face'],
                    self.circle_radius['face']
                )
            
            # 绘制手部
            for hand_type in ['left_hand', 'right_hand']:
                if pose_data.get(hand_type):
                    self._draw_landmarks(
                        image,
                        pose_data[hand_type],
                        self.colors['hand'],
                        self.thickness['hand'],
                        self.circle_radius['hand']
                    )
            
            return image
            
        except Exception as e:
            print(f"绘制姿态错误: {str(e)}")
            return image
    
    def _draw_landmarks(self, image, landmarks, color, thickness, radius):
        """绘制关键点和连接线"""
        h, w = image.shape[:2]
        
        # 绘制关键点
        for lm in landmarks:
            x, y = int(lm['x'] * w), int(lm['y'] * h)
            cv2.circle(image, (x, y), radius, color, -1)
            
            # 如果有可见度信息，显示在关键点旁边
            if 'visibility' in lm and lm['visibility'] > 0.5:
                cv2.putText(
                    image,
                    f"{lm['visibility']:.2f}",
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    color,
                    1
                )
        
        # 绘制连接线
        for i in range(len(landmarks) - 1):
            x1, y1 = int(landmarks[i]['x'] * w), int(landmarks[i]['y'] * h)
            x2, y2 = int(landmarks[i+1]['x'] * w), int(landmarks[i+1]['y'] * h)
            cv2.line(image, (x1, y1), (x2, y2), color, thickness) 
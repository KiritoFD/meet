import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import Dict, List, Tuple, Optional, Union

# Add missing import for PoseData
from .types import PoseData, Keypoint

from config.settings import POSE_CONFIG

logger = logging.getLogger(__name__)

class PoseDrawer:
    """姿态绘制器"""
    
    def __init__(self):
        """初始化绘制器"""
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
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
        
        logger.info("PoseDrawer 初始化完成")
        
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
            
    def draw(self, frame, results):
        """绘制姿态关键点和连接线"""
        try:
            if results.pose_landmarks:
                logger.info("开始绘制姿态")
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
                logger.info("姿态绘制完成")
                return True
            return False
        except Exception as e:
            logger.error(f"绘制姿态时出错: {str(e)}")
            return False

    def draw_pose(self, image: np.ndarray, pose_data: PoseData) -> np.ndarray:
        """
        在图像上绘制姿态关键点和连接线
        
        Args:
            image: 输入图像
            pose_data: 姿态数据
            
        Returns:
            绘制了姿态的图像
        """
        if image is None:
            logger.error("输入图像为空")
            return None
            
        if pose_data is None:
            logger.warning("没有姿态数据可绘制")
            return image
            
        # 获取关键点 - 同时支持keypoints和landmarks属性
        keypoints = getattr(pose_data, 'keypoints', None) or getattr(pose_data, 'landmarks', [])
        
        if not keypoints:
            logger.warning("姿态数据中没有关键点")
            return image
        
        try:
            # 创建一个副本以避免修改原始图像
            img_copy = image.copy()
            h, w = img_copy.shape[:2]
            
            # 转换关键点为MediaPipe格式以便使用内置绘制函数
            landmarks = self._convert_keypoints_to_landmarks(keypoints, h, w)
            
            # 创建结果对象，供MediaPipe绘制函数使用
            results = type('obj', (object,), {
                'pose_landmarks': landmarks,
                'pose_world_landmarks': None  # 不需要世界坐标系
            })
            
            # 使用MediaPipe绘制函数
            self.mp_drawing.draw_landmarks(
                img_copy,
                results.pose_landmarks,
                self.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            return img_copy
            
        except Exception as e:
            logger.error(f"绘制姿态时出错: {e}")
            return image
    
    def _convert_keypoints_to_landmarks(self, keypoints: List[Keypoint], height: int, width: int):
        """将我们的关键点格式转换为MediaPipe的landmarks格式"""
        
        landmarks_proto = self.mp_pose.PoseLandmarkList()
        
        for kp in keypoints:
            landmark = landmarks_proto.landmark.add()
            landmark.x = kp.x 
            landmark.y = kp.y
            landmark.z = kp.z if hasattr(kp, 'z') else 0.0
            landmark.visibility = kp.visibility if hasattr(kp, 'visibility') else 1.0
        
        return landmarks_proto
    
    def draw_comparison(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        绘制两张图像的比较视图
        
        Args:
            img1: 第一张图像
            img2: 第二张图像
            
        Returns:
            合并后的图像
        """
        if img1 is None or img2 is None:
            logger.error("输入图像为空")
            return None
            
        # 确保两张图像大小相同
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        if h1 != h2 or w1 != w2:
            img2 = cv2.resize(img2, (w1, h1))
            
        # 水平拼接
        return np.hstack((img1, img2))
    
    def draw_skeleton(self, image: np.ndarray, keypoints: List[Keypoint]) -> np.ndarray:
        """
        仅绘制骨架，不使用MediaPipe的绘制函数
        
        Args:
            image: 输入图像
            keypoints: 关键点列表
            
        Returns:
            绘制了骨架的图像
        """
        if image is None:
            return None
            
        if not keypoints:
            return image
            
        img_copy = image.copy()
        h, w = img_copy.shape[:2]
        
        # 绘制关键点
        for kp in keypoints:
            x, y = int(kp.x * w), int(kp.y * h)
            visibility = getattr(kp, 'visibility', 1.0)
            
            # 根据可见度调整颜色
            if visibility > 0.5:
                color = self.keypoint_color
            else:
                color = (0, 0, 255)  # 红色表示低可见度
                
            cv2.circle(img_copy, (x, y), self.radius, color, -1)
        
        # 绘制连接线
        for connection in self.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start_kp = keypoints[start_idx]
                end_kp = keypoints[end_idx]
                
                start_x, start_y = int(start_kp.x * w), int(start_kp.y * h)
                end_x, end_y = int(end_kp.x * w), int(end_kp.y * h)
                
                # 只绘制两个端点都可见的连接
                if getattr(start_kp, 'visibility', 1.0) > 0.5 and getattr(end_kp, 'visibility', 1.0) > 0.5:
                    cv2.line(img_copy, (start_x, start_y), (end_x, end_y), self.connection_color, self.thickness)
        
        return img_copy
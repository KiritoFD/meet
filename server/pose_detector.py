import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PoseLandmark:
    x: float
    y: float
    z: float
    visibility: float

@dataclass
class PoseResult:
    landmarks: List[PoseLandmark]
    frame_processed: np.ndarray
    success: bool
    pose_type: Optional[str] = None
    confidence: float = 0.0

class PoseDetector:
    def __init__(self, 
                 static_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """初始化姿态检测器"""
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # 姿态类型定义
        self.pose_types = {
            "t_pose": self._check_t_pose,
            "a_pose": self._check_a_pose,
            "natural": self._check_natural_pose
        }
        
        # 关节角度缓存
        self.joint_angles_buffer: List[Dict[str, float]] = []
        self.buffer_size = 5
        
    def detect(self, frame: np.ndarray) -> PoseResult:
        """检测单帧图像中的姿态"""
        try:
            # 转换颜色空间
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 进行姿态检测
            results = self.pose.process(frame_rgb)
            
            if not results.pose_landmarks:
                return PoseResult([], frame, False)
                
            # 提取关键点数据
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append(PoseLandmark(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z,
                    visibility=landmark.visibility
                ))
                
            # 在图像上绘制骨骼
            self.mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            
            # 识别姿态类型
            pose_type, confidence = self._identify_pose_type(landmarks)
            
            return PoseResult(landmarks, frame, True, pose_type, confidence)
            
        except Exception as e:
            logger.error(f"姿态检测失败: {e}")
            return PoseResult([], frame, False)
        
    def get_pose_type(self, landmarks: List[PoseLandmark]) -> Optional[str]:
        """识别当前姿态类型"""
        pose_type, _ = self._identify_pose_type(landmarks)
        return pose_type
        
    def calculate_joint_angles(self, landmarks: List[PoseLandmark]) -> dict:
        """计算关节角度"""
        angles = {}
        
        # TODO: 实现以下关节角度计算
        # 1. 肩关节角度
        # 2. 肘关节角度
        # 3. 髋关节角度
        # 4. 膝关节角度
        
        # 添加到缓冲区
        self.joint_angles_buffer.append(angles)
        if len(self.joint_angles_buffer) > self.buffer_size:
            self.joint_angles_buffer.pop(0)
            
        return self._smooth_angles(angles)
        
    def validate_pose(self, landmarks: List[PoseLandmark], pose_type: str) -> Tuple[bool, str]:
        """验证姿态是否符合要求"""
        if pose_type not in self.pose_types:
            return False, "未知的姿态类型"
            
        check_func = self.pose_types[pose_type]
        is_valid, confidence = check_func(landmarks)
        
        if not is_valid:
            return False, f"姿态不符合{pose_type}要求"
            
        return True, "姿态有效"
        
    def extract_measurements(self, landmarks: List[PoseLandmark]) -> dict:
        """从姿态数据中提取人体测量数据"""
        # TODO: 实现以下测量
        # 1. 身高估计
        # 2. 肩宽测量
        # 3. 臂长测量
        # 4. 腿长测量
        # 5. 躯干长度测量
        return {
            "height": 0,
            "shoulder_width": 0,
            "arm_length": 0,
            "leg_length": 0,
            "torso_length": 0
        }
        
    def optimize_landmarks(self, landmarks: List[PoseLandmark]) -> List[PoseLandmark]:
        """优化关键点数据，减少抖动"""
        # TODO: 实现以下优化
        # 1. 卡尔曼滤波
        # 2. 移动平均
        # 3. 异常值检测和处理
        return landmarks
        
    def _identify_pose_type(self, landmarks: List[PoseLandmark]) -> Tuple[Optional[str], float]:
        """识别姿态类型"""
        max_confidence = 0.0
        identified_type = None
        
        for pose_type, check_func in self.pose_types.items():
            is_valid, confidence = check_func(landmarks)
            if is_valid and confidence > max_confidence:
                max_confidence = confidence
                identified_type = pose_type
                
        return identified_type, max_confidence
        
    def _check_t_pose(self, landmarks: List[PoseLandmark]) -> Tuple[bool, float]:
        """检查是否为T-pose"""
        # TODO: 实现T-pose检测算法
        return False, 0.0
        
    def _check_a_pose(self, landmarks: List[PoseLandmark]) -> Tuple[bool, float]:
        """检查是否为A-pose"""
        # TODO: 实现A-pose检测算法
        return False, 0.0
        
    def _check_natural_pose(self, landmarks: List[PoseLandmark]) -> Tuple[bool, float]:
        """检查是否为自然站姿"""
        # TODO: 实现自然站姿检测算法
        return False, 0.0
        
    def _calculate_angle(self, p1: PoseLandmark, p2: PoseLandmark, p3: PoseLandmark) -> float:
        """计算三个点形成的角度"""
        # 转换为numpy数组
        v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
        
        # 计算角度
        cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        
        return np.degrees(angle)
        
    def _smooth_angles(self, angles: dict) -> dict:
        """平滑关节角度数据"""
        if not self.joint_angles_buffer:
            return angles
            
        smoothed_angles = {}
        for joint in angles:
            values = [frame[joint] for frame in self.joint_angles_buffer if joint in frame]
            if values:
                smoothed_angles[joint] = sum(values) / len(values)
                
        return smoothed_angles
        
    def serialize_landmarks(self, landmarks: List[PoseLandmark]) -> dict:
        """序列化关键点数据用于传输"""
        return {
            "landmarks": [
                {
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility
                }
                for lm in landmarks
            ]
        }
        
    def deserialize_landmarks(self, data: dict) -> List[PoseLandmark]:
        """反序列化关键点数据"""
        return [
            PoseLandmark(
                x=lm["x"],
                y=lm["y"],
                z=lm["z"],
                visibility=lm["visibility"]
            )
            for lm in data["landmarks"]
        ] 
import logging
from typing import Optional, Dict
import numpy as np
import time

# 按依赖顺序导入
from .types import PoseData
from .detector import PoseDetector
from .pose_binding import PoseBinding
from .pose_deformer import PoseDeformer
from .deform_smoother import DeformSmoother
from .drawer import PoseDrawer

logger = logging.getLogger(__name__)

class PosePipeline:
    """姿态处理管线，协调各组件工作"""
    
    def __init__(self, config: Dict = None):
        """初始化处理管线
        
        Args:
            config: 配置字典，用于初始化各组件
        """
        # 初始化组件
        self.detector = PoseDetector()
        self.binder = PoseBinding()
        self.deformer = PoseDeformer()
        self.drawer = PoseDrawer()
        
        # 从配置初始化平滑器
        smoother_config = {}
        if config and 'smoother' in config:
            smoother_config = config['smoother']
        self.smoother = DeformSmoother(**smoother_config)
        
        # 处理状态
        self._last_frame = None
        self._last_pose = None
        self._regions = {}
        
    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """处理单帧图像
        
        Args:
            frame: 输入图像帧
            
        Returns:
            处理后的图像帧，如果处理失败则返回None
        """
        try:
            # 1. 姿态检测
            detection = self.detector.detect(frame)
            if detection is None:
                logger.warning("姿态检测失败")
                return None
                
            # 2. 提取姿态数据
            landmarks = self.detector.mediapipe_to_keypoints(
                detection['pose'].pose_landmarks
            )
            pose_data = PoseData(landmarks=landmarks)
            
            # 3. 处理姿态数据
            if self.smoother:
                pose_data = self.smoother.smooth(pose_data)
                
            # 4. 存储结果
            self._last_pose = pose_data
            
            # 5. 绘制结果
            if self.drawer:
                frame = self.drawer.draw_frame(
                    frame=frame,
                    pose_results=detection['pose'],
                    hands_results=detection['hands'],
                    face_results=detection['face_mesh']
                )
                
            return frame
            
        except Exception as e:
            logger.error(f"处理失败: {e}")
            return None
            
    def reset(self):
        """重置处理状态"""
        self._last_frame = None
        self._last_pose = None
        self._regions.clear()
        self.smoother.reset()  # 重要：重置平滑器状态
        
    def release(self):
        """释放资源"""
        self.detector.release() 
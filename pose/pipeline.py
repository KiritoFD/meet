import logging
from typing import Optional, Dict
import numpy as np

# 按依赖顺序导入
from .pose_data import PoseData
from .detector import PoseDetector
from .binding import PoseBinder
from .pose_deformer import PoseDeformer
from .deform_smoother import DeformSmoother

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
        self.binder = PoseBinder()
        self.deformer = PoseDeformer()
        
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
            frame: 输入图像
            
        Returns:
            处理后的图像，失败返回None
        """
        try:
            # 1. 姿态检测
            detection = self.detector.detect(frame)
            if detection is None:
                logger.warning("姿态检测失败")
                return None
                
            # 2. 提取姿态数据
            pose_data = PoseData(
                keypoints=self.detector.mediapipe_to_keypoints(
                    detection['pose'].pose_landmarks
                )
            )
            
            # 3. 生成/更新绑定
            if not self._regions:
                self._regions = self.binder.generate_binding_points(
                    frame.shape,
                    pose_data
                )
            
            # 4. 应用变形
            deformed = self.deformer.deform_frame(
                frame=frame,
                regions=self._regions,
                target_pose=pose_data
            )
            
            # 5. 应用平滑
            smoothed = self.smoother.smooth_deformed_frame(
                frame=deformed,      # 变形后的帧
                original=frame,      # 原始帧
                pose=pose_data      # 姿态数据
            )
            
            # 6. 质量评估和自适应调整
            quality, scores = self.smoother.assess_quality(smoothed)
            if quality < 0.7:
                logger.warning(f"处理质量不佳: {quality:.2f}")
                logger.debug(f"详细评分: {scores}")
                
                # 根据具体问题调整参数
                if scores['temporal'] < 0.6:  # 时间不连续
                    self.smoother.temporal_weight = min(
                        self.smoother.temporal_weight + 0.1,
                        0.8
                    )
                if scores['edge'] < 0.6:  # 边缘问题
                    self.smoother.edge_width = min(
                        self.smoother.edge_width + 1,
                        5
                    )
            
            # 更新状态
            self._last_frame = frame.copy()
            self._last_pose = pose_data
            
            return smoothed
            
        except Exception as e:
            logger.error(f"处理失败: {str(e)}")
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
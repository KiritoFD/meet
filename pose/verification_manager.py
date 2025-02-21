import cv2
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from .pose_binding import PoseBinding
from .pose_detector import PoseDetector
from .pose_types import PoseData

logger = logging.getLogger(__name__)

@dataclass
class VerificationResult:
    """验证结果数据类"""
    success: bool
    message: str
    confidence: float = 0.0
    reference_data: Optional[Dict] = None

class VerificationManager:
    """身份验证管理器"""
    
    def __init__(self):
        """初始化验证管理器"""
        self.detector = PoseDetector()
        self.binder = PoseBinding()
        self.reference_data = None
        self.verified = False
        
    def capture_reference(self, frame: np.ndarray) -> VerificationResult:
        """捕获参考帧
        
        Args:
            frame: 输入图像帧
            
        Returns:
            VerificationResult: 验证结果
        """
        try:
            # 0. 验证输入
            if frame is None or frame.size == 0 or len(frame.shape) != 3:
                return VerificationResult(
                    success=False,
                    message="无效的输入图像"
                )

            # 1. 检测姿态
            pose_data = self.detector.detect(frame)
            if not pose_data:
                return VerificationResult(
                    success=False,
                    message="未检测到人物姿态，请确保人物完整出现在画面中"
                )
            
            # 2. 检查姿态关键点
            if len(pose_data.landmarks) < 33:
                return VerificationResult(
                    success=False,
                    message=f"检测到的关键点不完整: {len(pose_data.landmarks)}/33，请调整姿势"
                )

            # 检查关键点可见度
            visible_points = [lm for lm in pose_data.landmarks if lm.visibility > 0.5]
            if len(visible_points) < 15:
                return VerificationResult(
                    success=False,
                    message=f"姿态检测置信度过低: {len(visible_points)}/33 个关键点可见，请调整位置和光线"
                )
            
            # 3. 检查面部关键点
            if not pose_data.face_landmarks or len(pose_data.face_landmarks) < 50:
                return VerificationResult(
                    success=False,
                    message="未检测到足够的面部特征点，请确保面部清晰可见"
                )
            
            # 4. 创建绑定区域
            regions = self.binder.create_binding(frame, pose_data)
            if not regions:
                return VerificationResult(
                    success=False,
                    message="无法创建有效的绑定区域，请调整姿势"
                )
            
            # 5. 保存参考数据
            self.reference_data = {
                'pose': pose_data,
                'regions': regions,
                'frame': frame.copy()
            }
            
            return VerificationResult(
                success=True,
                message="参考帧捕获成功",
                confidence=1.0,
                reference_data=self.reference_data
            )
            
        except Exception as e:
            logger.error(f"参考帧捕获失败: {str(e)}")
            return VerificationResult(
                success=False,
                message=f"参考帧捕获出错: {str(e)}"
            )

    def verify_identity(self, frame: np.ndarray) -> VerificationResult:
        """验证身份
        
        Args:
            frame: 输入图像帧
            
        Returns:
            VerificationResult: 验证结果
        """
        try:
            # 1. 检查是否有参考数据
            if not self.reference_data:
                return VerificationResult(
                    success=False,
                    message="未找到参考数据，请先捕获参考帧"
                )
            
            # 2. 检测当前帧姿态
            pose_data = self.detector.detect(frame)
            if not pose_data:
                return VerificationResult(
                    success=False,
                    message="未检测到人物姿态"
                )
                
            # 3. 计算相似度
            similarity = self._calculate_similarity(
                self.reference_data['pose'],
                pose_data
            )
            
            # 4. 判断验证结果
            if similarity >= 0.85:  # 设置较高的阈值
                self.verified = True
                return VerificationResult(
                    success=True,
                    message="身份验证通过",
                    confidence=similarity
                )
            else:
                return VerificationResult(
                    success=False,
                    message="身份验证失败：姿态差异过大",
                    confidence=similarity
                )
                
        except Exception as e:
            logger.error(f"身份验证失败: {str(e)}")
            return VerificationResult(
                success=False,
                message=f"验证过程出错: {str(e)}"
            )
            
    def _calculate_similarity(self, ref_pose: PoseData, 
                            current_pose: PoseData) -> float:
        """计算两个姿态的相似度"""
        try:
            # 提取面部关键点
            ref_points = np.array([
                [lm['x'], lm['y']] for lm in ref_pose.face_landmarks
            ])
            current_points = np.array([
                [lm['x'], lm['y']] for lm in current_pose.face_landmarks
            ])
            
            # 计算特征点距离
            if len(ref_points) != len(current_points):
                return 0.0
                
            distances = np.linalg.norm(ref_points - current_points, axis=1)
            similarity = 1.0 / (1.0 + np.mean(distances))
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"相似度计算失败: {str(e)}")
            return 0.0
            
    def is_verified(self) -> bool:
        """检查是否已通过验证"""
        return self.verified
        
    def reset(self):
        """重置验证状态"""
        self.reference_data = None
        self.verified = False

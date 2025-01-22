import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import logging
import face_recognition
from dataclasses import dataclass

@dataclass
class FaceVerificationResult:
    """人脸验证结果"""
    is_same_person: bool
    confidence: float
    error_message: Optional[str] = None

class FaceVerifier:
    """人脸验证器"""
    
    def __init__(self, 
                 min_face_size: Tuple[int, int] = (30, 30),
                 similarity_threshold: float = 0.6,
                 max_angle: float = 30):
        """
        初始化人脸验证器
        
        Args:
            min_face_size: 最小人脸尺寸
            similarity_threshold: 相似度阈值
            max_angle: 最大角度差异
        """
        self.min_face_size = min_face_size
        self.similarity_threshold = similarity_threshold
        self.max_angle = max_angle
        self.reference_encoding = None
        self.logger = logging.getLogger(__name__)

    def set_reference(self, frame: np.ndarray) -> bool:
        """
        设置参考帧
        
        Args:
            frame: 参考帧图像
            
        Returns:
            bool: 是否成功设置参考帧
        """
        try:
            # 检测人脸
            face_locations = face_recognition.face_locations(frame)
            if not face_locations:
                self.logger.warning("No face detected in reference frame")
                return False
                
            # 获取人脸特征
            self.reference_encoding = face_recognition.face_encodings(
                frame, 
                face_locations
            )[0]
            
            self.logger.info("Reference frame set successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting reference frame: {str(e)}")
            return False

    def verify_face(self, frame: np.ndarray) -> FaceVerificationResult:
        """
        验证当前帧与参考帧是否为同一人
        
        Args:
            frame: 当前视频帧
            
        Returns:
            FaceVerificationResult: 验证结果
        """
        if self.reference_encoding is None:
            return FaceVerificationResult(
                is_same_person=False,
                confidence=0.0,
                error_message="No reference face set"
            )
            
        try:
            # 检测人脸
            face_locations = face_recognition.face_locations(frame)
            if not face_locations:
                return FaceVerificationResult(
                    is_same_person=False,
                    confidence=0.0,
                    error_message="No face detected in current frame"
                )
                
            # 获取人脸特征
            face_encoding = face_recognition.face_encodings(frame, face_locations)[0]
            
            # 计算相似度
            face_distances = face_recognition.face_distance(
                [self.reference_encoding], 
                face_encoding
            )
            similarity = 1 - face_distances[0]
            
            # 判断是否为同一人
            is_same = similarity >= self.similarity_threshold
            
            return FaceVerificationResult(
                is_same_person=is_same,
                confidence=float(similarity)
            )
            
        except Exception as e:
            self.logger.error(f"Error during face verification: {str(e)}")
            return FaceVerificationResult(
                is_same_person=False,
                confidence=0.0,
                error_message=str(e)
            )

    def verify_face_async(self, frame: np.ndarray) -> FaceVerificationResult:
        """
        异步验证人脸(用于实时视频流)
        
        Args:
            frame: 当前视频帧
            
        Returns:
            FaceVerificationResult: 验证结果
        """
        # 降低分辨率以提高性能
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # 转换为RGB
        rgb_small_frame = small_frame[:, :, ::-1]
        
        return self.verify_face(rgb_small_frame)

    def get_face_location(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        获取人脸位置
        
        Args:
            frame: 图像帧
            
        Returns:
            Optional[Tuple[int, int, int, int]]: 人脸位置(top, right, bottom, left)
        """
        try:
            face_locations = face_recognition.face_locations(frame)
            return face_locations[0] if face_locations else None
        except Exception as e:
            self.logger.error(f"Error getting face location: {str(e)}")
            return None

    def draw_face_box(self, frame: np.ndarray, 
                     result: FaceVerificationResult) -> np.ndarray:
        """
        在图像上绘制人脸框和验证结果
        
        Args:
            frame: 图像帧
            result: 验证结果
            
        Returns:
            np.ndarray: 绘制结果图像
        """
        face_location = self.get_face_location(frame)
        if face_location is None:
            return frame
            
        output = frame.copy()
        top, right, bottom, left = face_location
        
        # 根据验证结果选择颜色
        color = (0, 255, 0) if result.is_same_person else (0, 0, 255)
        
        # 绘制人脸框
        cv2.rectangle(output, (left, top), (right, bottom), color, 2)
        
        # 添加文本信息
        text = f"Match: {result.confidence:.2f}"
        cv2.putText(output, text, (left, top - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return output

    def clear_reference(self):
        """清除参考帧"""
        self.reference_encoding = None
        self.logger.info("Reference frame cleared") 
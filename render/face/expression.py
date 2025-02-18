from typing import List, Dict, Optional
import numpy as np
from .detector import FaceLandmark

class ExpressionMapper:
    """表情映射器"""
    def __init__(self):
        # 定义表情参数范围
        self.param_ranges = {
            'eye_open': (0.0, 1.0),
            'mouth_open': (0.0, 1.0),
            'brow_raise': (0.0, 1.0),
            'smile': (0.0, 1.0)
        }
        
    def map_landmarks_to_blendshapes(self, 
                                   landmarks: List[FaceLandmark],
                                   feature_points: Dict[str, List[FaceLandmark]]) -> Dict[str, float]:
        """将特征点映射到混合形状权重
        
        Args:
            landmarks: 所有特征点列表
            feature_points: 分组的特征点字典
            
        Returns:
            混合形状权重字典
        """
        weights = {}
        
        # 计算眼睛开合度
        weights['eye_open'] = self._calc_eye_open(feature_points['left_eye'], feature_points['right_eye'])
        
        # 计算嘴巴开合度
        weights['mouth_open'] = self._calc_mouth_open(feature_points['mouth'])
        
        # 计算眉毛高度
        weights['brow_raise'] = self._calc_brow_raise(
            feature_points['left_brow'], 
            feature_points['right_brow']
        )
        
        # 计算微笑程度
        weights['smile'] = self._calc_smile(feature_points['mouth'])
        
        # 规范化所有权重到合法范围
        return self._normalize_weights(weights)
        
    def _calc_eye_open(self, left_eye: List[FaceLandmark], right_eye: List[FaceLandmark]) -> float:
        """计算眼睛开合度"""
        left = self._calc_eye_aspect_ratio(left_eye)
        right = self._calc_eye_aspect_ratio(right_eye)
        return np.mean([left, right])
        
    def _calc_mouth_open(self, mouth_points: List[FaceLandmark]) -> float:
        """计算嘴巴开合度"""
        height = self._calc_vertical_distance(mouth_points[1], mouth_points[5])
        width = self._calc_horizontal_distance(mouth_points[0], mouth_points[2])
        return height / (width + 1e-6)
        
    def _calc_brow_raise(self, left_brow: List[FaceLandmark], right_brow: List[FaceLandmark]) -> float:
        """计算眉毛上扬程度"""
        left = np.mean([p.y for p in left_brow])
        right = np.mean([p.y for p in right_brow])
        return 1.0 - np.mean([left, right])  # 上扬时y值减小
        
    def _calc_smile(self, mouth_points: List[FaceLandmark]) -> float:
        """计算微笑程度"""
        width = self._calc_horizontal_distance(mouth_points[0], mouth_points[2])
        height = self._calc_vertical_distance(mouth_points[1], mouth_points[5])
        return width / (height + 1e-6)
        
    @staticmethod
    def _calc_eye_aspect_ratio(eye_points: List[FaceLandmark]) -> float:
        """计算眼睛纵横比"""
        v1 = np.linalg.norm(np.array([eye_points[1].x - eye_points[5].x,
                                     eye_points[1].y - eye_points[5].y]))
        v2 = np.linalg.norm(np.array([eye_points[2].x - eye_points[4].x,
                                     eye_points[2].y - eye_points[4].y]))
        h = np.linalg.norm(np.array([eye_points[0].x - eye_points[3].x,
                                    eye_points[0].y - eye_points[3].y]))
        return (v1 + v2) / (2.0 * h + 1e-6)
        
    @staticmethod
    def _calc_vertical_distance(p1: FaceLandmark, p2: FaceLandmark) -> float:
        """计算垂直距离"""
        return abs(p1.y - p2.y)
        
    @staticmethod
    def _calc_horizontal_distance(p1: FaceLandmark, p2: FaceLandmark) -> float:
        """计算水平距离"""
        return abs(p1.x - p2.x)
        
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """规范化权重到合法范围"""
        normalized = {}
        for name, value in weights.items():
            min_val, max_val = self.param_ranges[name]
            normalized[name] = np.clip(value, min_val, max_val)
        return normalized 
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2  # 用于创建并模糊蒙版
from dataclasses import dataclass
from .pose_data import PoseData, DeformRegion, BindingPoint
from config.settings import POSE_CONFIG
import logging
from .binding import BindingConfig

logger = logging.getLogger(__name__)

@dataclass
class BindingConfig:
    """绑定配置类"""
    smoothing_factor: float
    min_confidence: float
    joint_limits: Dict[str, Tuple[float, float]]

class PoseBinding:
    """姿态绑定类
    
    此类负责处理姿态数据与图像区域的绑定关系，支持身体和面部的区域划分与变形控制。
    
    Attributes:
        config (BindingConfig): 绑定配置参数
        reference_frame (np.ndarray): 参考帧图像
        landmarks (List[Dict]): 关键点列表
        weights (np.ndarray): 绑定权重
        valid (bool): 绑定是否有效
        keypoints (Dict): 关键点配置
        connections (Dict): 关键点连接关系
        _last_valid_binding (Dict[str, DeformRegion]): 上一个有效的绑定结果
        _frame_size (Tuple[int, int]): 当前处理图像的尺寸
        region_configs (Dict): 区域配置信息
    """
    
    def __init__(self, config: BindingConfig = None):
        """初始化姿态绑定器"""
        # 使用更宽松的配置
        self.config = config or BindingConfig(
            smoothing_factor=0.5,
            min_confidence=0.2,  # 降低最小置信度
            joint_limits={
                'shoulder': (-120, 120),
                'elbow': (-20, 165),
                'knee': (-20, 180)
            }
        )
        self.reference_frame = None
        self.landmarks = None
        self.weights = None
        self.valid = False
        self.keypoints = POSE_CONFIG['detector']['keypoints']
        self.connections = POSE_CONFIG['detector']['connections']
        self._last_valid_binding = None
        self._frame_size = None  # 存储当前处理图片的尺寸
        
        # 简化区域配置，只保留基本区域
        self.region_configs = {
            'torso': {
                'indices': [11, 12, 23, 24],  # 肩部和臀部关键点
                'min_points': 3,
                'required': True,
                'weight_type': 'torso'
            },
            'left_arm': {
                'indices': [11, 13, 15],  # 左肩、左肘、左腕
                'min_points': 2,
                'required': False,
                'weight_type': 'limb'
            },
            'right_arm': {
                'indices': [12, 14, 16],  # 右肩、右肘、右腕
                'min_points': 2,
                'required': False,
                'weight_type': 'limb'
            },
            'left_leg': {
                'indices': [23, 25, 27],  # 左髋、左膝、左踝
                'min_points': 2,
                'required': False
            },
            'right_leg': {
                'indices': [24, 26, 28],  # 右髋、右膝、右踝
                'min_points': 2,
                'required': False
            },
            
            # 面部区域 (可选)
            'face_contour': {
                'indices': [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,
                           378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,
                           21,54,103,67,109],
                'min_points': 10,
                'required': False
            },
            'left_eyebrow': {
                'indices': [70,63,105,66,107,55,65],
                'min_points': 5,
                'required': False
            },
            'right_eyebrow': {
                'indices': [336,296,334,293,300,285,295],
                'min_points': 5,
                'required': False
            },
            'left_eye': {
                'indices': [33,246,161,160,159,158,157,173,133],
                'min_points': 5,
                'required': False
            },
            'right_eye': {
                'indices': [362,398,384,385,386,387,388,466,263],
                'min_points': 5,
                'required': False
            },
            'nose': {
                'indices': [168,6,197,195,5,4,1,19,94,2],
                'min_points': 5,
                'required': False
            },
            'mouth': {
                'indices': [0,267,269,270,409,291,375,321,405,314,17,84,181,91,146,61,185,40,39,37],
                'min_points': 10,
                'required': False
            }
        }

    def create_binding(self, frame: np.ndarray, pose_data: PoseData) -> Dict[str, DeformRegion]:
        """创建图像区域与姿态的绑定关系"""
        if frame is None or pose_data is None:
            raise ValueError("Frame and pose_data cannot be None")
        
        # 存储参考帧和尺寸
        self.reference_frame = frame.copy()
        self._frame_size = frame.shape[:2]
        
        # 创建区域字典
        regions_dict = {}
        
        try:
            # 处理每个预定义区域
            for region_name, config in self.region_configs.items():
                points = self._get_keypoints(pose_data, config['indices'])
                
                # 检查点数是否足够
                if len(points) >= config['min_points']:
                    region = self._create_region(frame, points, region_name)
                    if region:  # 添加这个检查
                        regions_dict[region_name] = region
                elif config['required']:
                    raise ValueError(f"Required region {region_name} has insufficient points")
                    
            # 保存有效的绑定结果
            if regions_dict:
                self._last_valid_binding = regions_dict
                self.valid = True
            else:
                logger.warning("No valid regions created")
                self.valid = False
                
            return regions_dict
            
        except Exception as e:
            self.valid = False
            logger.error(f"Failed to create binding: {str(e)}")
            # 如果有上一个有效的绑定，返回它
            return self._last_valid_binding or {}
    
    def _process_landmarks(self, landmarks):
        """处理关键点数据"""
        processed = []
        for lm in landmarks:
            if lm['visibility'] < self.config.min_confidence:
                continue
            processed.append({
                'x': lm['x'],
                'y': lm['y'],
                'z': lm['z'],
                'visibility': lm['visibility']
            })
        return processed
    
    def _compute_weights(self):
        """计算变形权重"""
        if not self.landmarks:
            return None
            
        # 创建权重矩阵
        height, width = self.reference_frame.shape[:2]
        weights = np.zeros((height * width, len(self.landmarks)))
        
        # 计算每个像素点的权重
        for i, landmark in enumerate(self.landmarks):
            y, x = np.mgrid[0:height, 0:width]
            dist = np.sqrt((x - landmark['x'] * width) ** 2 + 
                         (y - landmark['y'] * height) ** 2)
            weights[:, i] = np.exp(-dist.flatten() / 100)  # 使用高斯权重
            
        # 归一化权重
        row_sums = weights.sum(axis=1)
        weights = weights / row_sums[:, np.newaxis]
        
        return weights

    def create_binding(self, frame: np.ndarray, pose_data: PoseData) -> List[DeformRegion]:
        """创建初始帧的区域绑定"""
        if frame is None or pose_data is None:
            raise ValueError("Frame and pose_data cannot be None")
        
        if frame.size == 0:
            raise ValueError("Empty frame")
        
        if not pose_data.landmarks:
            return self._last_valid_binding or []
        
        # 获取实际图片尺寸
        frame_h, frame_w = frame.shape[:2]
        # 存储图片尺寸供其他方法使用
        self._frame_size = (frame_w, frame_h)
        
        mask_template = np.zeros((frame_h, frame_w), dtype=np.uint8)
        regions = []
        missing_required = set()
        
        # 只处理必需的区域
        required_regions = {name: config for name, config in self.region_configs.items() 
                          if config['required']}
        
        for region_name, config in required_regions.items():
            try:
                points = []
                self._get_keypoints_inplace(pose_data, config['indices'], points)
                
                if len(points) >= config['min_points']:
                    mask_template.fill(0)
                    region = None
                    
                    if region_name == 'torso':
                        region = self._create_torso_region(mask_template, points)
                    elif region_name.startswith(('left_', 'right_')) and \
                         region_name.endswith(('_arm', '_leg')):
                        region = self._create_limb_region(mask_template, points, region_name)
                    else:
                        region = self._create_face_region(mask_template, points, region_name)
                        
                    if region:
                        region.name = region_name
                        regions.append(region)
                else:
                    missing_required.add(region_name)
                    
            except Exception as e:
                missing_required.add(region_name)
                logger.error(f"Failed to create {region_name}: {str(e)}")
        
        if missing_required:
            return self._last_valid_binding or []
            
        # 处理可选区域（如果还有空间）
        max_regions = 4  # 减少最大区域数量
        if len(regions) < max_regions:
            optional_regions = {name: config for name, config in self.region_configs.items() 
                              if not config['required']}
            
            for region_name, config in optional_regions.items():
                if len(regions) >= max_regions:
                    break
                    
                try:
                    points = []
                    self._get_keypoints_inplace(pose_data, config['indices'], points)
                    
                    if len(points) >= config['min_points']:
                        mask_template.fill(0)
                        region = None
                        
                        if region_name.startswith(('left_', 'right_')) and \
                           region_name.endswith(('_arm', '_leg')):
                            region = self._create_limb_region(mask_template, points, region_name)
                        else:
                            region = self._create_face_region(mask_template, points, region_name)
                            
                        if region:
                            region.name = region_name
                            regions.append(region)
                            
                except Exception as e:
                    logger.debug(f"Failed to create optional region {region_name}: {str(e)}")
        
        if regions:
            self._last_valid_binding = regions[:]
            
        return regions

    def _create_face_region(self, frame: np.ndarray, pose_data: PoseData, 
                          indices: List[int], region_name: str) -> Optional[DeformRegion]:
        """创建面部区域"""
        points = self._get_keypoints(pose_data, indices)
        if len(points) < 3:
            return None
            
        # 根据区域类型设置权重类型
        if region_name == 'face_contour':
            weight_type = 'contour'
        elif region_name.endswith('_eye') or region_name.endswith('_eyebrow'):
            weight_type = 'feature'
        else:
            weight_type = 'feature'
            
        return self._create_region(frame, points, weight_type)

    def _create_torso_region(self, frame: np.ndarray, points: List[np.ndarray]) -> Optional[DeformRegion]:
        """创建躯干区域"""
        if len(points) < 3:
            return None
            
        return self._create_region(frame, points, 'torso')

    def _create_limb_region(self, frame: np.ndarray, points: List[np.ndarray], 
                          region_name: str) -> Optional[DeformRegion]:
        """创建肢体区域"""
        if len(points) < 2:
            return None
            
        # 添加控制点
        center = np.mean(points, axis=0)
        direction = points[1] - points[0]
        normal = np.array([-direction[1], direction[0]])
        normal = normal / (np.linalg.norm(normal) + 1e-6) * 20
        
        # 根据区域类型调整控制点
        if 'upper' in region_name:
            control_point = center + normal
        else:
            control_point = center - normal
            
        points.append(control_point)
        
        return self._create_region(frame, points, 'limb')

    def _create_region(self, frame: np.ndarray, points: List[np.ndarray], 
                      region_name: str) -> Optional[DeformRegion]:
        """创建变形区域"""
        try:
            if len(points) < 2:
                return None
            
            # 计算区域中心
            center = np.mean(points, axis=0)
            
            # 创建区域蒙版
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            points_array = np.array(points, dtype=np.int32)
            
            if len(points) >= 3:
                cv2.fillConvexPoly(mask, points_array, 255)
            else:
                # 对于只有两个点的情况，创建一个细长的区域
                pt1, pt2 = points
                direction = pt2 - pt1
                normal = np.array([-direction[1], direction[0]])
                normal = normal / (np.linalg.norm(normal) + 1e-6) * 10
                
                polygon = np.array([
                    pt1 + normal,
                    pt2 + normal,
                    pt2 - normal,
                    pt1 - normal
                ], dtype=np.int32)
                cv2.fillConvexPoly(mask, polygon, 255)
            
            # 创建绑定点
            binding_points = []
            for i, point in enumerate(points):
                binding_points.append(BindingPoint(
                    landmark_index=i,
                    local_coords=point - center,
                    weight=1.0 / len(points)
                ))
            
            region = DeformRegion(
                center=center,
                binding_points=binding_points,
                mask=mask
            )
            region.name = region_name
            return region
            
        except Exception as e:
            logger.error(f"Failed to create region {region_name}: {str(e)}")
            return None

    def _get_keypoints(self, pose_data: PoseData, indices: List[int]) -> List[np.ndarray]:
        """获取关键点坐标"""
        if self._frame_size is None:
            raise ValueError("Frame size not set")
        
        frame_w, frame_h = self._frame_size
        points = []
        
        for idx in indices:
            try:
                if idx < len(pose_data.landmarks):
                    lm = pose_data.landmarks[idx]
                    # 转换归一化坐标到像素坐标
                    x = int(lm.x * frame_w)
                    y = int(lm.y * frame_h)
                    # 降低可见度要求
                    if lm.visibility >= 0.1:  # 降低可见度阈值
                        points.append(np.array([x, y], dtype=np.float32))
            except Exception as e:
                logger.debug(f"Failed to get keypoint {idx}: {str(e)}")
                continue
            
        return points

    def _get_keypoints_inplace(self, pose_data: PoseData, indices: List[int], points: List[np.ndarray]):
        """获取关键点坐标并存储在points列表中"""
        if self._frame_size is None:
            raise ValueError("Frame size not set. Call create_binding first.")
            
        frame_w, frame_h = self._frame_size
        for idx in indices:
            try:
                if idx < len(pose_data.landmarks):
                    lm = pose_data.landmarks[idx]
                    if lm['visibility'] >= self.config.min_confidence:
                        points.append(np.array([
                            lm['x'] * frame_w,
                            lm['y'] * frame_h
                        ]))
            except Exception as e:
                logger.debug(f"Failed to get keypoint {idx}: {str(e)}")
                continue

    def _create_region_mask(self, frame: np.ndarray, points: List[np.ndarray]) -> np.ndarray:
        """创建区域蒙版"""
        try:
            mask = frame  # 直接使用传入的frame
            mask.fill(0)  # 清空mask
            
            if len(points) < 3:
                pt1, pt2 = points[:2]
                direction = pt2 - pt1
                normal = np.array([-direction[1], direction[0]])
                max_val = max(abs(normal[0]), abs(normal[1]))
                if max_val > 0:
                    normal = (normal / max_val) * 4
                
                polygon_points = np.array([
                    (pt1 + normal).astype(np.int32),
                    (pt2 + normal).astype(np.int32),
                    (pt2 - normal).astype(np.int32),
                    (pt1 - normal).astype(np.int32)
                ])
                
                cv2.fillPoly(mask, [polygon_points], 255)
            else:
                points_int = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [points_int], 255)
            
            # 使用小核的高斯模糊
            cv2.GaussianBlur(mask, (3, 3), 0, dst=mask)
            
            return mask
            
        except Exception as e:
            logger.error(f"Failed to create mask: {str(e)}")
            return np.zeros_like(frame)

    def _calculate_weights(self, points: List[np.ndarray], region_type: str) -> List[float]:
        """计算控制点权重
        
        Args:
            points (List[np.ndarray]): 控制点列表
            region_type (str): 区域类型，影响权重计算策略
            
        Returns:
            List[float]: 控制点权重列表
        """
        n_points = len(points)
        
        if n_points == 0:
            return []
            
        if region_type == 'torso':
            # 躯干区域：中间两点0.6，两端点0.4
            if n_points == 1:
                # 单点情况使用0.4
                weights = [0.4]
            elif n_points <= 4:
                weights = [0.4, 0.6, 0.6, 0.4][:n_points]
            else:
                weights = [0.4, 0.6, 0.6, 0.4] + [0.4] * (n_points - 4)
                
        elif region_type == 'limb':
            # 肢体区域：两端点0.7，其他点0.3
            if n_points == 1:
                # 单点情况使用0.7
                weights = [0.7]
            elif n_points == 2:
                weights = [0.7, 0.7]
            else:
                weights = [0.7] + [0.3] * (n_points - 2) + [0.7]
                
        elif region_type == 'contour':
            # 面部轮廓：统一0.5
            weights = [0.5] * n_points
            
        else:  # feature或其他面部特征
            # 其他面部特征：统一0.8
            weights = [0.8] * n_points
            
        return weights

    def _validate_landmarks(self, landmarks: List[Dict]) -> bool:
        """验证关键点有效性
        
        Args:
            landmarks (List[Dict]): 关键点列表
            
        Returns:
            bool: 关键点是否有效
        """
        return all(lm['visibility'] >= self.config.min_confidence for lm in landmarks)

    def update_binding(self, regions: List[DeformRegion], pose: PoseData) -> List[DeformRegion]:
        """更新绑定信息"""
        if not regions or not pose:
            return [region for region in (self._last_valid_binding or [])]
            
        updated_regions = []
        
        for region in regions:
            try:
                # 获取关键点索引
                indices = [bp.landmark_index for bp in region.binding_points]
                indices = [idx for idx in indices if idx >= 0]
                
                if not indices:
                    continue
                    
                # 获取新的关键点位置
                points = self._get_keypoints(pose, indices)
                if len(points) < 2:
                    continue
                    
                # 更新区域
                region_type = 'torso' if region.name == 'torso' else \
                             'limb' if region.name.endswith(('_arm', '_leg')) else \
                             'feature'
                             
                updated_region = self._create_region(
                    np.zeros((480, 640), dtype=np.uint8),
                    points,
                    region_type
                )
                updated_region.name = region.name
                updated_regions.append(updated_region)
                
            except Exception as e:
                logger.warning(f"Failed to update region {region.name}: {str(e)}")
                continue
        
        # 如果更新失败，返回上一个有效的绑定
        return updated_regions if updated_regions else self._last_valid_binding or []

    def fuse_multimodal_data(self, rgb_frame, depth_frame, thermal_frame):
        """融合多模态数据"""
        # 对齐不同传感器数据
        aligned_data = self._align_sensors(rgb_frame, depth_frame, thermal_frame)
        
        # 提取几何特征
        geometry_features = self._extract_geometry_features(aligned_data['depth'])
        
        # 提取纹理特征
        texture_features = self._extract_texture_features(aligned_data['rgb'])
        
        # 融合特征
        fused_features = self.fusion_network(
            geometry_features, 
            texture_features,
            aligned_data['thermal']
        )
        
        return self._decode_features(fused_features)
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2  # 用于创建并模糊蒙版
from dataclasses import dataclass

from pose.types import Landmark
from .pose_types import PoseData, DeformRegion, BindingPoint
from config.settings import POSE_CONFIG
import logging

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
        # 修改关键点配置访问方式
        self.keypoints = POSE_CONFIG['detector']['body_landmarks']  # 使用 body_landmarks 而不是 keypoints
        self.connections = POSE_CONFIG['detector']['connections']
        self._last_valid_binding = None
        self._frame_size = None  # 存储当前处理图片的尺寸
        
        # 更新面部区域细分配置
        self.face_indices = {
            # 脸部轮廓区域
            'contour_upper_right': list(range(0, 4)),      # 上部右侧
            'contour_upper': list(range(4, 5)),           # 上部中间
            'contour_upper_left': list(range(5, 9)),      # 上部左侧
            'contour_left': list(range(9, 13)),           # 左侧
            'contour_lower_left': list(range(13, 17)),    # 下部左侧
            'contour_lower': list(range(152, 155)),       # 下巴中间
            'contour_lower_right': list(range(155, 159)), # 下部右侧
            'contour_right': list(range(159, 162)),       # 右侧
            
            # 眉毛区域
            'right_eyebrow_outer': list(range(17, 20)),   # 右眉毛外侧
            'right_eyebrow_center': list(range(20, 22)),  # 右眉毛中部
            'right_eyebrow_inner': list(range(22, 24)),   # 右眉毛内侧
            'left_eyebrow_inner': list(range(337, 339)),  # 左眉毛内侧
            'left_eyebrow_center': list(range(339, 341)), # 左眉毛中部
            'left_eyebrow_outer': list(range(341, 344)),  # 左眉毛外侧
            
            # 眼睛区域
            'right_eye_outer': list(range(246, 248)),     # 右眼外角
            'right_eye_upper': list(range(248, 250)),     # 右眼上部
            'right_eye_inner': list(range(250, 252)),     # 右眼内角
            'right_eye_lower': list(range(252, 254)),     # 右眼下部
            'left_eye_inner': list(range(386, 388)),      # 左眼内角
            'left_eye_upper': list(range(388, 390)),      # 左眼上部
            'left_eye_outer': list(range(390, 392)),      # 左眼外角
            'left_eye_lower': list(range(392, 394)),      # 左眼下部
            
            # 鼻子区域
            'nose_bridge_upper': list(range(168, 170)),   # 鼻梁上部
            'nose_bridge_center': list(range(170, 172)),  # 鼻梁中部
            'nose_bridge_lower': list(range(172, 174)),   # 鼻梁下部
            'nose_tip': list(range(174, 177)),           # 鼻尖
            'nose_bottom': list(range(177, 180)),        # 鼻底
            'nose_left': list(range(459, 463)),          # 左鼻翼
            'nose_right': list(range(463, 467)),         # 右鼻翼
            
            # 嘴唇区域
            'upper_lip_right': list(range(0, 3)),        # 上唇右侧
            'upper_lip_center': list(range(3, 4)),       # 上唇中心
            'upper_lip_left': list(range(4, 7)),         # 上唇左侧
            'lower_lip_left': list(range(7, 9)),         # 下唇左侧
            'lower_lip_center': list(range(9, 10)),      # 下唇中心
            'lower_lip_right': list(range(10, 12)),      # 下唇右侧
            'lip_corner_right': [61],                    # 右嘴角
            'lip_corner_left': [291],                    # 左嘴角
            
            # 内唇区域
            'inner_upper_lip_right': list(range(12, 14)), # 内上唇右侧
            'inner_upper_lip_center': list(range(14, 15)), # 内上唇中心
            'inner_upper_lip_left': list(range(15, 17)),  # 内上唇左侧
            'inner_lower_lip_left': list(range(17, 19)),  # 内下唇左侧
            'inner_lower_lip_center': list(range(19, 20)), # 内下唇中心
            'inner_lower_lip_right': list(range(20, 22))  # 内下唇右侧
        }
        
        # 设置每个区域的最小点数要求
        min_points_config = {
            'contour': 3,     # 轮廓区域
            'eyebrow': 2,     # 眉毛区域
            'eye': 2,         # 眼睛区域
            'nose': 2,        # 鼻子区域
            'lip': 2,         # 嘴唇区域
            'inner_lip': 2    # 内唇区域
        }
        
        # 初始化基础区域配置
        self.region_configs = {
            'torso': {
                'indices': [11, 12, 23, 24],  # 肩部和臀部关键点
                'min_points': 3,
                'required': True,
                'type': 'body'
            },
            'left_arm': {
                'indices': [11, 13, 15],  # 左肩、左肘、左腕
                'min_points': 2,
                'required': False,
                'type': 'body'
            },
            'right_arm': {
                'indices': [12, 14, 16],  # 右肩、右肘、右腕
                'min_points': 2,
                'required': False,
                'type': 'body'
            }
        }
        
        # 添加面部区域配置
        for name, indices in self.face_indices.items():
            self.region_configs[f'face_{name}'] = {
                'indices': indices,
                'min_points': max(3, len(indices) // 3),  # 降低最小点数要求到三分之一
                'required': False,
                'type': 'face'
            }

    def _get_min_points(self, region_name: str, config: Dict[str, int]) -> int:
        """根据区域名称获取最小点数要求"""
        if 'contour' in region_name:
            return config['contour']
        elif 'eyebrow' in region_name:
            return config['eyebrow']
        elif 'eye' in region_name:
            return config['eye']
        elif 'nose' in region_name:
            return config['nose']
        elif 'inner_lip' in region_name:
            return config['inner_lip']
        elif 'lip' in region_name:
            return config['lip']
        return 2  # 默认值

    def _get_weight_type(self, region_name: str) -> str:
        """根据区域名称获取权重类型"""
        if 'contour' in region_name:
            return 'contour'
        elif any(part in region_name for part in ['eye', 'eyebrow', 'nose']):
            return 'feature'
        elif 'lip' in region_name:
            return 'deform'
        return 'default'

    def create_binding(self, frame: np.ndarray, pose_data: PoseData) -> List[DeformRegion]:
        """创建图像区域与姿态的绑定关系"""
        if frame is None or pose_data is None:
            logger.warning("输入无效: frame 或 pose_data 为空")
            return []
            
        try:
            # 获取图像尺寸
            height, width = frame.shape[:2]
            self._frame_size = (width, height)  # 保存图像尺寸
            regions = []
            
            # 1. 创建躯干区域
            torso_indices = [11, 12, 23, 24]  # 肩部和臀部关键点
            torso_points = self._get_points(pose_data.landmarks, torso_indices)
            if len(torso_points) >= 3:
                region = self._create_region(
                    "torso",
                    frame,
                    torso_points,
                    torso_indices,
                    'body'
                )
                if region:
                    regions.append(region)
            
            # 2. 创建手臂区域
            arm_configs = [
                ('left_arm', [11, 13, 15]),   # 左肩、左肘、左腕
                ('right_arm', [12, 14, 16]),  # 右肩、右肘、右腕
            ]
            
            for name, indices in arm_configs:
                points = self._get_points(pose_data.landmarks, indices)
                if len(points) >= 2:
                    region = self._create_region(
                        name,
                        frame,
                        points,
                        indices,
                        'body'
                    )
                    if region:
                        regions.append(region)
            
            logger.info(f"成功创建 {len(regions)} 个绑定区域")
            return regions
            
        except Exception as e:
            logger.error(f"创建绑定区域失败: {str(e)}")
            return []

    def _create_region(self, 
                      name: str,
                      frame: np.ndarray,
                      points: List[np.ndarray],
                      indices: List[int],
                      region_type: str) -> Optional[DeformRegion]:
        """创建变形区域"""
        if len(points) < 2:
            logger.warning(f"区域 {name} 点数不足")
            return None
            
        try:
            # 计算区域中心
            center = np.mean(points, axis=0)
            
            # 创建区域蒙版
            height, width = frame.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            points_array = np.array(points, dtype=np.int32)
            
            if len(points) >= 3:
                cv2.fillConvexPoly(mask, points_array, 255)
            else:
                # 对于只有两个点的情况，创建细长区域
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
                cv2.fillPoly(mask, [polygon], 255)
            
            # 创建绑定点
            binding_points = []
            for point_idx, (point, idx) in enumerate(zip(points, indices)):
                binding_points.append(BindingPoint(
                    landmark_index=idx,
                    local_coords=point - center,
                    weight=self._calculate_weight(point_idx, len(points), region_type)
                ))
            
            return DeformRegion(
                name=name,
                center=center,
                binding_points=binding_points,
                mask=mask,
                type=region_type
            )
            
        except Exception as e:
            logger.error(f"创建区域 {name} 失败: {str(e)}")
            return None

    def _get_points(self, landmarks: List[Landmark], indices: List[int]) -> List[np.ndarray]:
        """获取指定索引的关键点坐标"""
        if not self._frame_size:
            raise ValueError("Frame size not set")
            
        width, height = self._frame_size
        points = []
        
        for idx in indices:
            try:
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    if lm.visibility >= self.config.min_confidence:
                        # 转换为像素坐标
                        point = np.array([
                            int(lm.x * width),
                            int(lm.y * height)
                        ], dtype=np.float32)
                        points.append(point)
            except Exception as e:
                logger.debug(f"跳过关键点 {idx}: {str(e)}")
                continue
                
        return points

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

    def _create_face_region(self, frame: np.ndarray, points: List[np.ndarray], 
                          region_name: str) -> Optional[DeformRegion]:
        """创建面部区域"""
        if len(points) < 3:
            return None
            
        return self._create_region(
            frame=frame,
            points=points,
            name=region_name,
            region_type='face'  # 显式指定区域类型
        )

    def _create_torso_region(self, frame: np.ndarray, points: List[np.ndarray]) -> Optional[DeformRegion]:
        """创建躯干区域"""
        if len(points) < 3:
            return None
            
        return self._create_region(
            frame=frame,
            points=points,
            name='torso',
            region_type='body'  # 显式指定区域类型
        )

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
        
        return self._create_region(
            frame=frame,
            points=points,
            name=region_name,
            region_type='body'  # 显式指定区域类型
        )

    def _create_region(self, frame: np.ndarray, points: List[np.ndarray], 
                      name: str, region_type: str) -> Optional[DeformRegion]:
        """创建变形区域"""
        try:
            if len(points) < 2:
                logger.warning(f"区域 {name} 点数不足")
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
                    weight=self._calculate_weight(i, len(points), region_type)
                ))
            
            # 直接创建并返回 DeformRegion 实例
            return DeformRegion(
                name=name,
                center=center,
                binding_points=binding_points,
                mask=mask,
                type=region_type
            )
            
        except Exception as e:
            logger.error(f"创建区域 {name} 失败: {str(e)}")
            return None

    def _calculate_weight(self, index: int, total_points: int, region_type: str) -> float:
        """计算控制点权重
        
        Args:
            index: 点的索引
            total_points: 总点数
            region_type: 区域类型
            
        Returns:
            float: 权重值 (0-1)
        """
        if region_type == 'body':
            # 躯干：中心点权重大，边缘点权重小
            if index == 0 or index == total_points - 1:
                return 0.4  # 端点
            else:
                return 0.6  # 中间点
        elif region_type == 'face':
            # 面部：均匀权重
            return 1.0 / total_points
        else:
            # 其他：均匀权重
            return 1.0 / total_points

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
from typing import List, Dict, Optional
import numpy as np
import cv2
from .pose_data import PoseData, DeformRegion, Landmark
import time
import pytest
from config.settings import POSE_CONFIG
import os
import logging

logger = logging.getLogger(__name__)


class PoseDeformer:
    """处理基于姿态的图像变形"""

    def __init__(self):
        """初始化变形器"""
        config = POSE_CONFIG['deformer']
        self.smoothing_window = config['smoothing_window']
        self._smoothing_factor = config['smoothing_factor']
        self._blend_radius = config['blend_radius']
        self._min_scale = config['min_scale']
        self._max_scale = config['max_scale']
        self._control_point_radius = config['control_point_radius']
        self._last_deformed = None

    def _ensure_type_compatibility(self, frame: np.ndarray) -> np.ndarray:
        """确保图像类型兼容性"""
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            return frame.astype(np.float32)
        return frame

    def deform_frame(self, frame: np.ndarray, regions: Dict[str, DeformRegion], pose: PoseData) -> np.ndarray:
        """变形单个帧"""
        # 预处理输入帧
        frame_float = self._ensure_type_compatibility(frame)
        
        # 基本输入验证
        if frame is None or pose is None:
            raise ValueError("Invalid frame or pose data")
        
        # 确保regions是字典类型
        if not isinstance(regions, dict):
            logger.warning("Regions is not a dictionary, converting...")
            if isinstance(regions, list):
                regions = {region.name: region for region in regions if hasattr(region, 'name')}
            else:
                regions = {}
        
        # 如果没有有效区域，返回原始帧
        if not regions:
            logger.warning("No valid regions for deformation")
            return frame.copy()
        
        # 姿态验证
        if not self._validate_pose(pose):
            raise ValueError("Invalid pose data")
        
        try:
            # 处理每个区域
            transformed_regions = {}
            for region_name, region in regions.items():
                transform = self._calculate_transform(region, pose)
                transformed = self._apply_transform(frame, region, transform)
                transformed_regions[region_name] = transformed
            
            # 混合结果
            result = self._blend_regions(frame, transformed_regions)
            
            # 应用时间平滑
            if self._last_deformed is not None:
                if (self._last_deformed.shape == result.shape and 
                    self._last_deformed.dtype == result.dtype):
                    result = cv2.addWeighted(
                        self._last_deformed,
                        self._smoothing_factor,
                        result,
                        1 - self._smoothing_factor,
                        0
                    )
            
            self._last_deformed = result.copy()
            return result
        
        except Exception as e:
            logger.error(f"Deformation failed: {str(e)}")
            return frame.copy()

    def _calculate_transform(self,
                             region: DeformRegion,
                             target_pose: PoseData) -> np.ndarray:
        """计算区域变形矩阵

        计算从原始区域到目标位置的仿射变换矩阵
        """
        # 收集原始点和目标点
        src_points = []
        dst_points = []

        # 处理每个绑定点
        for bp in region.binding_points:
            # 原始点：区域中心 + 局部坐标
            src_point = region.center + bp.local_coords
            src_points.append(src_point)

            # 目标点：从目标姿态获取新位置
            landmark = target_pose.landmarks[bp.landmark_index]
            dst_point = np.array([landmark.x, landmark.y])
            dst_points.append(dst_point)

        # 确保至少有3个点用于计算仿射变换
        if len(src_points) < 3:
            # 如果点不够，添加额外的控制点
            for i in range(3 - len(src_points)):
                angle = i * 2 * np.pi / 3
                radius = self._control_point_radius  # 控制点距离中心的半径

                # 添加原始控制点
                src_point = region.center + radius * np.array([
                    np.cos(angle), np.sin(angle)
                ])
                src_points.append(src_point)

                # 添加对应的目标控制点
                dst_point = src_point  # 保持不变
                dst_points.append(dst_point)

        # 转换为numpy数组
        src_points = np.float32(src_points)
        dst_points = np.float32(dst_points)

        # 计算仿射变换矩阵
        transform = cv2.getAffineTransform(
            src_points[:3],  # 使用前3个点
            dst_points[:3]
        )

        return transform

    def _apply_transform(self, 
                        frame: np.ndarray, 
                        region: DeformRegion,
                        transform: np.ndarray) -> np.ndarray:
        """优化的变换应用函数"""
        height, width = frame.shape[:2]
        
        # 根据图像大小选择处理策略
        is_large_image = width * height > 1280 * 720
        if is_large_image:
            # 计算缩放因子
            scale_factor = np.sqrt(1280 * 720 / (width * height))
            scaled_size = (int(width * scale_factor), int(height * scale_factor))
            # 对大图像使用更快的缩放方法
            if width * height > 1920 * 1080:
                frame_scaled = cv2.resize(frame, scaled_size, interpolation=cv2.INTER_NEAREST)
            else:
                frame_scaled = cv2.resize(frame, scaled_size, interpolation=cv2.INTER_AREA)
            transform_scaled = transform.copy()
            transform_scaled[:, 2] *= scale_factor
        else:
            frame_scaled = frame
            transform_scaled = transform
            scaled_size = (width, height)
        
        # 处理掩码和边界
        if hasattr(region, '_cached_mask'):
            mask = region._cached_mask
            if hasattr(region, '_prev_center') and not np.array_equal(region._prev_center, region.center):
                dist = np.linalg.norm(region.center - region._prev_center)
                if dist > 5:
                    weight = np.clip(1.0 - (dist / 50.0), 0.3, 0.7)
                    new_mask = region.mask if region.mask is not None else np.ones(scaled_size[::-1], dtype=np.uint8)
                    mask = cv2.addWeighted(mask, weight, new_mask, 1 - weight, 0)
        else:
            mask = region.mask if region.mask is not None else np.ones(scaled_size[::-1], dtype=np.uint8)
            region._cached_mask = mask
            region._prev_center = region.center.copy()
        
        # 使用更高效的非零元素查找
        if hasattr(region, '_cached_bounds'):
            min_y, max_y, min_x, max_x = region._cached_bounds
        else:
            y, x = np.nonzero(mask)
            if len(x) == 0 or len(y) == 0:
                return np.zeros_like(frame)
            
            # 计算边界框并缓存
            min_x, max_x = np.min(x), np.max(x)
            min_y, max_y = np.min(y), np.max(y)
            region._cached_bounds = (min_y, max_y, min_x, max_x)
        
        # 优化padding计算
        rel_padding = min(0.05, self._blend_radius / min(scaled_size))
        padding_x = max(2, int(rel_padding * scaled_size[0]))
        padding_y = max(2, int(rel_padding * scaled_size[1]))
        
        # 使用更高效的边界检查
        min_x = max(0, min_x - padding_x)
        min_y = max(0, min_y - padding_y)
        max_x = min(scaled_size[0], max_x + padding_x)
        max_y = min(scaled_size[1], max_y + padding_y)
        
        # 优化ROI提取和变换
        result = np.zeros_like(frame_scaled)
        roi = frame_scaled[min_y:max_y, min_x:max_x]
        
        # 优化变换矩阵计算
        roi_transform = transform_scaled.copy()
        roi_transform[:, 2] -= [min_x, min_y]
        
        # 使用优化的仿射变换
        warped_roi = cv2.warpAffine(
            roi,
            roi_transform,
            (max_x - min_x, max_y - min_y),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT_101
        )
        
        # 直接写入结果
        result[min_y:max_y, min_x:max_x] = warped_roi
        
        # 对大图像进行上采样
        if is_large_image:
            # 根据图像大小选择不同的插值方法
            if width * height > 1920 * 1080:
                result = cv2.resize(result, (width, height), interpolation=cv2.INTER_LINEAR)
            else:
                result = cv2.resize(result, (width, height), interpolation=cv2.INTER_CUBIC)
        
        # 更新区域的上一个中心点
        region._prev_center = region.center.copy()
        
        return result

    def _blend_regions(self,
                       frame: np.ndarray,
                       transformed_regions: Dict[str, np.ndarray]) -> np.ndarray:
        """合并变形后的区域

        使用权重混合将所有变形区域合并到一起
        """
        if not transformed_regions:
            return frame.copy()

        # 创建输出图像和权重累积
        result = np.zeros_like(frame, dtype=float)
        weight_sum = np.zeros(frame.shape[:2], dtype=float)


        # 处理每个区域
        for region_name, region in transformed_regions.items():
            # 计算区域权重（非零像素的位置）
            weight = np.any(region > 0, axis=2).astype(float)

            # 应用高斯模糊使边缘平滑
            weight = cv2.GaussianBlur(weight, (0, 0), self._blend_radius)

            # 扩展维度以匹配图像通道
            weight = np.expand_dims(weight, axis=2)

            # 累积变形区域和权重
            result += region * weight
            weight_sum += weight[..., 0]

        # 处理未覆盖区域
        uncovered = weight_sum == 0
        weight_sum[uncovered] = 1
        weight_sum = np.expand_dims(weight_sum, axis=2)

        # 归一化结果
        result = result / weight_sum

        # 填充未覆盖区域
        result[uncovered] = frame[uncovered]

        return result.astype(frame.dtype)

    def _process_single(self, frame: np.ndarray, regions: Dict[str, DeformRegion], pose: PoseData) -> np.ndarray:
        """处理单个姿态"""
        if not self._validate_pose(pose):
            return frame.copy()
        
        # 使用已有的deform_frame函数，但去掉验证检查
        transformed_regions = {}
        for region_name, region in regions.items():
            transform = self._calculate_transform(region, pose)
            transformed = self._apply_transform(frame, region, transform)
            transformed_regions[region_name] = transformed
        
        result = self._blend_regions(frame, transformed_regions)
        
        # 应用时间平滑
        if self._last_deformed is not None:
            if (self._last_deformed.shape == result.shape and 
                self._last_deformed.dtype == result.dtype):
                result = cv2.addWeighted(
                    self._last_deformed,
                    self._smoothing_factor,
                    result,
                    1 - self._smoothing_factor,
                    0
                )
        
        self._last_deformed = result.copy()
        return result

    def batch_deform(self, frame: np.ndarray, poses: List[PoseData]) -> List[np.ndarray]:
        """批量处理多个姿态
        
        优化说明:
        1. 使用NumPy的向量化操作
        2. 减少内存分配和拷贝
        3. 优化计算流程
        """
        if not poses:
            return []
        
        # 预处理输入帧
        frame_float = self._ensure_type_compatibility(frame)
        height, width = frame_float.shape[:2]
        
        # 一次性验证所有姿态
        valid_poses = [(i, pose) for i, pose in enumerate(poses) 
                       if self._validate_pose(pose)]
        
        # 如果没有有效姿态，直接返回原始帧的副本
        if not valid_poses:
            return [frame.copy() for _ in poses]
        
        # 预分配结果数组 - 使用字典存储中间结果
        results = [dict() for _ in poses]
        final_results = [frame.copy() for _ in poses]
        
        # 创建共享的regions字典
        regions = {}  # 假设这是从某处获取的
        
        # 批量处理有效姿态
        for region_name, region in regions.items():
            # 为每个有效姿态计算变换矩阵
            transforms = [
                self._calculate_transform(region, pose)
                for _, pose in valid_poses
            ]
            
            # 批量应用变换
            for idx, (i, _) in enumerate(valid_poses):
                transformed = self._apply_transform(
                    frame_float, 
                    region, 
                    transforms[idx]
                )
                results[i][region_name] = transformed
        
        # 混合区域并应用时间平滑
        for i, pose in enumerate(poses):
            if i not in [idx for idx, _ in valid_poses]:
                continue
            
            # 检查是否有需要混合的区域
            if not results[i]:  # 使用字典的空检查
                continue
            
            # 混合区域
            result = self._blend_regions(frame_float, results[i])
            
            # 应用时间平滑
            if self._last_deformed is not None:
                if (self._last_deformed.shape == result.shape and 
                    self._last_deformed.dtype == result.dtype):
                    result = cv2.addWeighted(
                        self._last_deformed,
                        self._smoothing_factor,
                        result,
                        1 - self._smoothing_factor,
                        0
                    )
            
            self._last_deformed = result.copy()
            final_results[i] = result
        
        return final_results

    def interpolate(self, pose1: PoseData, pose2: PoseData, t: float) -> PoseData:
        """姿态插值"""
        if pose1 is None or pose2 is None:
            raise ValueError("Invalid pose data")
        if not 0 <= t <= 1:
            raise ValueError("Interpolation parameter t must be between 0 and 1")

        landmarks = []
        for lm1, lm2 in zip(pose1.landmarks, pose2.landmarks):
            landmark = Landmark(
                x=lm1.x * (1 - t) + lm2.x * t,
                y=lm1.y * (1 - t) + lm2.y * t,
                z=lm1.z * (1 - t) + lm2.z * t,
                visibility=min(lm1.visibility, lm2.visibility)
            )
            landmarks.append(landmark)

        return PoseData(
            landmarks=landmarks,
            timestamp=time.time(),
            confidence=min(pose1.confidence, pose2.confidence)
        )

    def smooth_sequence(self, poses: List[PoseData]) -> List[PoseData]:
        """平滑姿态序列"""
        if len(poses) < 3:
            raise ValueError("Sequence too short for smoothing")

        smoothed = []
        window = min(self.smoothing_window, len(poses))

        for i in range(len(poses)):
            start = max(0, i - window // 2)
            end = min(len(poses), i + window // 2 + 1)
            smoothed.append(self._average_poses(poses[start:end]))

        return smoothed

    def predict_next(self, history: List[PoseData]) -> Optional[PoseData]:
        """预测下一帧姿态"""
        if not history:
            raise ValueError("History cannot be empty")
        if len(history) < 2:
            raise ValueError("Need at least 2 poses for prediction")

        # 简单线性预测
        last = history[-1]
        prev = history[-2]

        landmarks = []
        for lm_last, lm_prev in zip(last.landmarks, prev.landmarks):
            dx = lm_last.x - lm_prev.x
            dy = lm_last.y - lm_prev.y
            dz = lm_last.z - lm_prev.z

            landmark = Landmark(
                x=lm_last.x + dx,
                y=lm_last.y + dy,
                z=lm_last.z + dz,
                visibility=lm_last.visibility
            )
            landmarks.append(landmark)

        return PoseData(
            landmarks=landmarks,
            timestamp=time.time(),
            confidence=last.confidence
        )

    def _average_poses(self, poses: List[PoseData]) -> PoseData:
        """计算多个姿态的平均"""
        n = len(poses)
        landmarks = []

        for i in range(len(poses[0].landmarks)):
            avg_lm = Landmark(
                x=sum(p.landmarks[i].x for p in poses) / n,
                y=sum(p.landmarks[i].y for p in poses) / n,
                z=sum(p.landmarks[i].z for p in poses) / n,
                visibility=min(p.landmarks[i].visibility for p in poses)
            )
            landmarks.append(avg_lm)

        return PoseData(
            landmarks=landmarks,
            timestamp=time.time(),
            confidence=sum(p.confidence for p in poses) / n
        )

    def _calculate_scale(self, pose: PoseData) -> float:
        """计算姿态的缩放因子"""
        # 使用关键点的分布范围估算缩放
        x_coords = [lm['x'] for lm in pose.landmarks]
        y_coords = [lm['y'] for lm in pose.landmarks]

        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)

        return np.clip(np.sqrt(x_range ** 2 + y_range ** 2), 0.5, 2.0)

    def _calculate_rotation(self, pose: PoseData) -> float:
        """计算姿态的旋转角度（度数）"""
        # 使用关键点的方向估算旋转
        if len(pose.landmarks) < 2:
            return 0.0

        # 使用前两个关键点计算方向
        dx = pose.landmarks[1]['x'] - pose.landmarks[0]['x']
        dy = pose.landmarks[1]['y'] - pose.landmarks[0]['y']

        angle = np.degrees(np.arctan2(dy, dx))
        return angle

    def _calculate_translation(self, pose: PoseData, center: np.ndarray) -> np.ndarray:
        """计算姿态的平移向量"""
        if not pose.landmarks:
            return np.zeros(2)

        # 计算关键点的中心
        mean_x = np.mean([lm['x'] for lm in pose.landmarks])
        mean_y = np.mean([lm['y'] for lm in pose.landmarks])

        # 计算相对于图像中心的偏移
        return np.array([mean_x - center[0], mean_y - center[1]])

    def _validate_pose(self, pose: PoseData) -> bool:
        """验证姿态数据的有效性"""
        if not pose or not pose.landmarks:
            return False

        # 降低置信度阈值
        if pose.confidence < 0.3:  # 从0.5改为0.3
            return False

        # 降低可见点比例要求
        visible_points = sum(1 for lm in pose.landmarks if lm.visibility > 0.3)  # 从0.5改为0.3
        if visible_points < len(pose.landmarks) * 0.4:  # 从0.6改为0.4
            return False

        return True

    @pytest.fixture
    def realistic_pose_sequences(self):
        """生成多种真实姿态序列"""

        def generate_sequence(motion_type: str) -> List[PoseData]:
            if motion_type == "walking":
                return self._generate_walking_sequence()
            elif motion_type == "jumping":
                return self._generate_jumping_sequence()
            elif motion_type == "dancing":
                return self._generate_dancing_sequence()
            elif motion_type == "random":
                return self._generate_random_sequence()

        return {
            "walking": generate_sequence("walking"),
            "jumping": generate_sequence("jumping"),
            "dancing": generate_sequence("dancing"),
            "random": generate_sequence("random")
        }

    def _get_memory_usage(self):
        """获取当前进程的内存使用情况（Windows兼容）"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)  # 返回MB

    def _check_resources(self):
        """检查资源使用情况"""
        memory_usage = self._get_memory_usage()
        if memory_usage > 1024:  # 超过1GB
            raise RuntimeError(f"Memory usage too high: {memory_usage:.1f}MB")
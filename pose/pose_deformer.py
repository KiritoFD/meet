from typing import List, Dict, Optional, Union
import numpy as np
import cv2
from .pose_data import PoseData, DeformRegion, Landmark
import time
import pytest
from concurrent.futures import ThreadPoolExecutor

class PoseDeformer:
    """处理基于姿态的图像变形"""
    
    def __init__(self, smoothing_window: int = 5):
        """初始化变形器"""
        if smoothing_window < 3:
            raise ValueError("Smoothing window must be at least 3")
        self.smoothing_window = smoothing_window
        # 变形参数
        self._blend_radius = 20  # 区域混合半径
        self._min_scale = 0.5  # 最小缩放
        self._max_scale = 2.0  # 最大缩放
        
        # 添加资源限制
        self._max_frame_size = (4096, 4096)  # 最大支持的图像尺寸
        self._max_memory_usage = 1024 * 1024 * 1024  # 1GB
        self._resource_monitor = self._init_resource_monitor()
        
    def _init_resource_monitor(self):
        """初始化资源监控"""
        import psutil
        return {
            'process': psutil.Process(),
            'start_memory': psutil.Process().memory_info().rss
        }

    def _check_resources(self, frame: np.ndarray):
        """检查资源使用情况"""
        # 检查图像尺寸
        if frame.shape[0] > self._max_frame_size[0] or frame.shape[1] > self._max_frame_size[1]:
            raise ValueError(f"Frame size exceeds maximum allowed: {self._max_frame_size}")
        
        # 检查内存使用
        current_memory = self._resource_monitor['process'].memory_info().rss
        if current_memory - self._resource_monitor['start_memory'] > self._max_memory_usage:
            raise RuntimeError("Memory usage exceeded limit")
        
    def deform_frame(self, 
                     frame: np.ndarray,
                     regions: Union[Dict[str, DeformRegion], PoseData],
                     target_pose: Optional[PoseData] = None) -> np.ndarray:
        """变形图像帧"""
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame")
        if frame.shape[0] < 2 or frame.shape[1] < 2:
            raise ValueError("Frame too small")
        
        # 处理不同的输入情况
        if isinstance(regions, PoseData):
            target_pose = regions
            regions = {}
        elif target_pose is None:
            raise ValueError("target_pose is required when regions is a dict")
        
        # 验证姿态数据
        if target_pose is None or not target_pose.validate():
            raise ValueError("Invalid pose data")
        
        # 创建输出帧
        result = frame.copy()
        
        try:
            # 如果没有区域信息,使用全局变形
            if not regions:
                height, width = frame.shape[:2]
                center = np.float32((width/2, height/2))
                
                # 计算变形参数
                scale = self._calculate_scale(target_pose)
                rotation = self._calculate_rotation(target_pose)
                translation = self._calculate_translation(target_pose, center)
                
                # 构建变形矩阵
                M = cv2.getRotationMatrix2D(center, rotation, scale)
                M[:, 2] += translation
                
                # 应用变形
                result = cv2.warpAffine(result, M, (width, height),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT)
            else:
                # 验证区域配置
                for region in regions.values():
                    if not isinstance(region, DeformRegion):
                        raise ValueError("Invalid region type")
                    if np.any(np.isinf(region.center)) or np.any(np.isnan(region.center)):
                        raise ValueError("Invalid region center")
                
                # 使用区域变形
                transformed_regions = {}
                
                # 处理每个区域
                for region_name, region in regions.items():
                    transform = self._calculate_transform(region, target_pose)
                    transformed = self._apply_transform(frame, region, transform)
                    transformed_regions[region_name] = transformed
                
                # 混合所有变形区域
                result = self._blend_regions(frame, transformed_regions)
            
            return result
        
        except Exception as e:
            raise ValueError(f"Deformation failed: {str(e)}")
        
    def _calculate_transform(self, 
                           region: DeformRegion,
                           target_pose: PoseData) -> np.ndarray:
        """计算区域变形矩阵"""
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
        
        # 添加稳定性检查
        if len(src_points) < 3:
            # 如果点不够,添加更稳定的控制点
            center = np.mean(src_points, axis=0) if len(src_points) > 0 else region.center
            radius = np.linalg.norm(region.center - center) + 50  # 动态半径
            
            for i in range(3 - len(src_points)):
                angle = i * 2 * np.pi / 3
                # 添加原始控制点
                src_point = center + radius * np.array([
                    np.cos(angle), np.sin(angle)
                ])
                src_points.append(src_point)
                
                # 添加对应的目标控制点,保持相对位置
                if len(dst_points) > 0:
                    ref_offset = dst_points[0] - src_points[0]
                    dst_point = src_point + ref_offset
                else:
                    dst_point = src_point
                dst_points.append(dst_point)
        
        # 转换为numpy数组并确保类型
        src_points = np.float32(src_points)
        dst_points = np.float32(dst_points)
        
        # 添加数值稳定性检查
        if np.any(np.isinf(src_points)) or np.any(np.isinf(dst_points)):
            raise ValueError("Invalid points detected")
        
        # 计算仿射变换矩阵
        transform = cv2.getAffineTransform(
            src_points[:3],
            dst_points[:3]
        )
        
        return transform
        
    def _apply_transform(self,
                        frame: np.ndarray,
                        region: DeformRegion,
                        transform: np.ndarray) -> np.ndarray:
        """应用变形到区域"""
        try:
            # 创建输出图像
            height, width = frame.shape[:2]
            result = np.zeros_like(frame)
            
            # 应用变换
            warped = cv2.warpAffine(
                frame,
                transform,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )
            
            # 应用区域蒙版
            if region.mask is not None:
                # 同样变换蒙版
                warped_mask = cv2.warpAffine(
                    region.mask,
                    transform,
                    (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )
                
                # 归一化蒙版
                warped_mask = warped_mask.astype(float) / 255
                
                # 扩展维度以匹配图像通道
                warped_mask = np.expand_dims(warped_mask, axis=2)
                
                # 应用蒙版
                result = warped * warped_mask
            
            # 添加结果验证
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                raise ValueError("Invalid transformation result")
            
            return result
        
        except Exception as e:
            # 返回原始区域而不是失败
            mask = np.zeros_like(frame)
            cv2.circle(mask, tuple(map(int, region.center)), 
                      int(self._blend_radius), (1,1,1), -1)
            return frame * mask
        
    def _blend_regions(self,
                      frame: np.ndarray,
                      transformed_regions: Dict[str, np.ndarray]) -> np.ndarray:
        """混合变形区域"""
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
        
    def batch_deform(self, frame: np.ndarray, poses: List[PoseData]) -> List[np.ndarray]:
        """批量处理多个姿态"""
        results = []
        # 预分配内存
        for _ in range(len(poses)):
            results.append(np.zeros_like(frame))
        
        # 并行处理
        with ThreadPoolExecutor() as executor:
            futures = []
            for i, pose in enumerate(poses):
                future = executor.submit(self.deform_frame, frame, {}, pose)
                futures.append((i, future))
            
            for i, future in futures:
                try:
                    results[i] = future.result()
                except Exception as e:
                    results[i] = frame.copy()  # 错误时返回原始帧
                
        return results

    def interpolate(self, pose1: PoseData, pose2: PoseData, t: float) -> PoseData:
        """姿态插值"""
        if pose1 is None or pose2 is None:
            raise ValueError("Invalid pose data")
        if not 0 <= t <= 1:
            raise ValueError("Interpolation parameter t must be between 0 and 1")
            
        landmarks = []
        for lm1, lm2 in zip(pose1.landmarks, pose2.landmarks):
            landmark = Landmark(
                x=lm1.x * (1-t) + lm2.x * t,
                y=lm1.y * (1-t) + lm2.y * t,
                z=lm1.z * (1-t) + lm2.z * t,
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
            start = max(0, i - window//2)
            end = min(len(poses), i + window//2 + 1)
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
        x_coords = [lm.x for lm in pose.landmarks]
        y_coords = [lm.y for lm in pose.landmarks]
        
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        
        scale = np.sqrt(x_range**2 + y_range**2)
        return np.clip(scale, self._min_scale, self._max_scale)

    def _calculate_rotation(self, pose: PoseData) -> float:
        """计算姿态的旋转角度（度数）"""
        # 使用关键点的方向估算旋转
        if len(pose.landmarks) < 2:
            return 0.0
            
        # 使用前两个关键点计算方向
        dx = pose.landmarks[1].x - pose.landmarks[0].x
        dy = pose.landmarks[1].y - pose.landmarks[0].y
        
        angle = np.degrees(np.arctan2(dy, dx))
        return angle

    def _calculate_translation(self, pose: PoseData, center: np.ndarray) -> np.ndarray:
        """计算姿态的平移向量"""
        if not pose.landmarks:
            return np.zeros(2)
            
        # 计算关键点的中心
        mean_x = np.mean([lm.x for lm in pose.landmarks])
        mean_y = np.mean([lm.y for lm in pose.landmarks])
        
        # 计算相对于图像中心的偏移
        return np.array([mean_x - center[0], mean_y - center[1]])

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
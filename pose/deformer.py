"""
姿态变形系统 (PoseDeformer)
==========================

该模块提供了实时姿态变形的核心功能，包括骨骼变换、网格变形和结果优化。

主要功能:
- 骨骼变换矩阵计算
- 网格点变形
- 抖动抑制
- 边界处理
- GPU加速支持

性能指标:
- 单帧处理时间 < 10ms
- GPU内存占用 < 500MB
- CPU使用率 < 20%

质量指标:
- 变形精度 > 90%
- 边缘锯齿 < 1px
- 纹理失真 < 5%

作者: [您的名字]
版本: 1.0.0
"""

from typing import List, Dict, Optional, Union, Tuple
import numpy as np
import torch
import cv2
from dataclasses import dataclass
from .binding import SkeletonBinding
import time
from concurrent.futures import ThreadPoolExecutor
import logging
import random
from collections import OrderedDict

logger = logging.getLogger(__name__)

class DeformationConfig:
    """变形配置类"""
    def __init__(self):
        # 基本配置
        self.batch_size = 1024
        self.use_gpu = True
        
        # 性能优化
        self.enable_adaptive_batch = True
        self.enable_cache = True
        self.cache_size = 1000
        
        # 质量控制
        self.smoothing_factor = 0.5
        self.jitter_threshold = 2.0
        self.boundary_padding = 10
        self.optimization_level = 1
        self.blend_weight = 0.7

class DeformationCache:
    """变形缓存类"""
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache = OrderedDict()
        
    def get(self, key: str) -> Optional[np.ndarray]:
        """获取缓存项"""
        if key in self._cache:
            value = self._cache.pop(key)  # 移除并重新插入以更新顺序
            self._cache[key] = value
            return value.copy()  # 返回副本以防止修改
        return None
        
    def put(self, key: str, value: np.ndarray):
        """添加缓存项"""
        if key in self._cache:
            self._cache.pop(key)
        elif len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)  # 移除最旧的项
        self._cache[key] = value.copy()  # 存储副本
        
    def __contains__(self, key: str) -> bool:
        return key in self._cache
        
    def clear(self):
        """清空缓存"""
        self._cache.clear()

class PoseDeformer:
    """
    姿态变形器
    
    该类实现了实时姿态变形的核心功能，支持GPU加速和性能监控。
    
    参数:
        binding (SkeletonBinding): 骨骼绑定对象，包含参考姿态和权重信息
        
    异常:
        ValueError: 当提供的骨骼绑定无效时抛出
    """
    
    def __init__(self, binding: SkeletonBinding):
        """初始化变形器"""
        if not binding.valid:
            raise ValueError("Invalid skeleton binding")
            
        self.binding = binding
        self.config = DeformationConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.config.use_gpu else 'cpu')
        
        # 初始化性能统计
        self.perf_stats = {
            'total_time': [],
            'batch_size': [],
            'memory_usage': []
        }
        
        # 初始化状态
        self.frame_count = 0
        self.previous_result = None
        self.batch_size = 1024  # 默认批处理大小
        self.min_batch_size = 256
        self.max_batch_size = 4096
        
        logger.info(f"Initialized deformer with device: {self.device}")
        
        # 初始化批处理相关
        self.batch_size_history = []
        
        # 性能监控
        self.perf_stats = {
            'transform_time': [],
            'deform_time': [],
            'smooth_time': [],
            'total_time': [],
            'memory_usage': [],
            'batch_sizes': []
        }
        
        # 初始化GPU
        self._init_gpu()
        
        # 创建线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.num_threads)
        
        logger.info("PoseDeformer initialized with config: %s", self.config)
    
    def _init_gpu(self):
        """初始化GPU相关资源"""
        if self.device.type == 'cuda':
            # 检查可用显存
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            if gpu_mem < self.config.memory_limit_mb * 1024 * 1024:
                logger.warning("GPU memory might be insufficient")
            
            self._prepare_gpu_buffers()
            logger.info("GPU acceleration enabled on device: %s", self.device)
    
    def _prepare_gpu_buffers(self):
        """准备GPU缓存"""
        if self.device.type == 'cuda':
            # 将网格点转换为张量并移至GPU
            self.mesh_points_gpu = torch.from_numpy(
                np.column_stack((self.binding.mesh_points, 
                               np.ones(len(self.binding.mesh_points))))
            ).float().to(self.device)
            
            # 将权重矩阵转换为张量并移至GPU
            self.weights_gpu = torch.from_numpy(
                self.binding.weights
            ).float().to(self.device)
            
            # 创建结果缓冲区
            self.result_buffer = torch.zeros(
                (self.binding.reference_frame.shape[0],
                 self.binding.reference_frame.shape[1], 3),
                device=self.device,
                dtype=torch.uint8
            )
    
    def _get_adaptive_batch_size(self) -> int:
        """动态调整批处理大小"""
        if not self.config.enable_adaptive_batch:
            return self.config.batch_size
        
        try:
            # 获取最近的处理时间
            recent_times = self.perf_stats.get('total_time', [])[-5:]
            if not recent_times:
                return self.config.batch_size
            
            # 计算平均处理时间
            avg_time = sum(recent_times) / len(recent_times)
            target_time = 0.005  # 5ms
            
            # 根据处理时间调整批处理大小
            if avg_time > target_time:
                new_batch = max(self.min_batch_size, int(self.config.batch_size * 0.8))
            else:
                new_batch = min(self.max_batch_size, int(self.config.batch_size * 1.2))
            
            return new_batch
            
        except Exception as e:
            logger.error("Error in adaptive batch sizing: %s", str(e))
            return self.config.batch_size
    
    def _optimize_transforms(self, transforms: Union[List[np.ndarray], torch.Tensor]) -> Union[List[np.ndarray], torch.Tensor]:
        """优化变换矩阵"""
        if self.config.optimization_level == 0:
            return transforms
            
        if isinstance(transforms, torch.Tensor):
            # 应用SVD分解优化
            u, s, v = torch.svd(transforms[:, :2, :2])
            transforms[:, :2, :2] = torch.bmm(u, v.transpose(1, 2))
            
            # 限制平移范围
            transforms[:, :2, 2].clamp_(-1.0, 1.0)
        else:
            # CPU版本的优化
            for i in range(len(transforms)):
                u, s, v = np.linalg.svd(transforms[i][:2, :2])
                transforms[i][:2, :2] = np.dot(u, v)
                transforms[i][:2, 2] = np.clip(transforms[i][:2, 2], -1.0, 1.0)
        
        return transforms
    
    def _parallel_deform(self, points: np.ndarray, weights: np.ndarray, 
                        transforms: List[np.ndarray]) -> np.ndarray:
        """并行处理变形"""
        chunks = np.array_split(points, self.config.num_threads)
        weight_chunks = np.array_split(weights, self.config.num_threads)
        
        futures = []
        for chunk_points, chunk_weights in zip(chunks, weight_chunks):
            future = self.thread_pool.submit(
                self._deform_chunk, chunk_points, chunk_weights, transforms)
            futures.append(future)
        
        results = []
        for future in futures:
            results.append(future.result())
        
        return np.concatenate(results)
    
    def _deform_chunk(self, points: np.ndarray, weights: np.ndarray, 
                      transforms: List[np.ndarray]) -> np.ndarray:
        """处理一个变形数据块"""
        result = np.zeros_like(points)
        for i in range(len(points)):
            transform = np.zeros((3, 3))
            for j, t in enumerate(transforms):
                transform += weights[i, j] * t
            
            point = np.append(points[i], 1)
            result[i] = np.dot(transform, point)[:2]
        
        return result
    
    def transform_frame(self, current_pose: List[Dict[str, float]]) -> np.ndarray:
        """
        变形单帧图像
        
        Args:
            current_pose: 当前姿态数据
            
        Returns:
            变形后的图像
        """
        try:
            start_time = time.time()
            
            # 验证输入
            if not self._validate_pose(current_pose):
                logger.error("Invalid pose data")
                return self.binding.reference_frame.copy()
            
            # 计算骨骼变换
            transforms = self.compute_bone_transforms(current_pose)
                    
            # 应用变形
            result = self.apply_deformation(transforms)
                
            # 后处理
            result = self._post_process(result)
            
            # 更新性能统计
            self._update_performance_stats(start_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Transform error: {str(e)}")
            return self.binding.reference_frame.copy()
    
    def _get_cache_key(self, pose_data: List[Dict[str, float]]) -> str:
        """
        生成姿态数据的缓存键
        
        Args:
            pose_data: 姿态数据列表
            
        Returns:
            缓存键字符串
        """
        try:
            # 将姿态数据转换为可哈希的形式
            key_parts = []
            for bone_data in pose_data:
                bone_key = []
                
                if 'rotation' in bone_data:
                    bone_key.append(('r', round(bone_data['rotation'], 3)))
                    
                if 'translation' in bone_data:
                    tx, ty = bone_data['translation']
                    bone_key.append(('t', round(tx, 3), round(ty, 3)))
                    
                if 'scale' in bone_data:
                    sx, sy = bone_data['scale']
                    bone_key.append(('s', round(sx, 3), round(sy, 3)))
                    
                key_parts.append(tuple(bone_key))
                
            # 生成唯一的字符串键
            return str(hash(tuple(key_parts)))
            
        except Exception as e:
            logger.error("Cache key generation error: %s", str(e))
            # 返回时间戳作为后备键，确保不会返回缓存的结果
            return str(time.time())
    
    def _post_process(self, result: np.ndarray) -> np.ndarray:
        """
        对变形结果进行后处理
        
        Args:
            result: 原始变形结果
            
        Returns:
            处理后的结果
        """
        try:
            # 应用基本平滑
            if self.config.smoothing_factor > 0:
                kernel_size = 3
                result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)
                
            # 简单的抖动抑制
            if self.previous_result is not None and self.config.jitter_threshold > 0:
                diff = np.abs(result.astype(np.float32) - self.previous_result.astype(np.float32))
                mask = diff < self.config.jitter_threshold
                result[mask] = self.previous_result[mask]
                
            # 更新状态
            self.previous_result = result.copy()
            self.frame_count += 1
            
            return result
            
        except Exception as e:
            logger.error("Post-processing error: %s", str(e))
            return result
    
    def _update_performance_stats(self, start_time: float):
        """更新性能统计数据"""
        try:
            # 计算处理时间
            total_time = time.time() - start_time
            self.perf_stats['total_time'].append(total_time)
            
            # 记录当前批处理大小
            current_batch = self.config.batch_size
            self.perf_stats['batch_size'].append(current_batch)
            
            # 记录GPU内存使用
            if self.device.type == 'cuda':
                memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                self.perf_stats['memory_usage'].append(memory)
            else:
                self.perf_stats['memory_usage'].append(0)
            
            # 限制历史数据大小
            max_history = 100
            for key in self.perf_stats:
                self.perf_stats[key] = self.perf_stats[key][-max_history:]
            
            # 更新批处理大小
            if self.config.enable_adaptive_batch:
                self.config.batch_size = self._get_adaptive_batch_size()
            
            self.frame_count += 1
            
        except Exception as e:
            logger.error(f"Error updating performance stats: {str(e)}")
    
    def get_performance_report(self) -> Dict[str, float]:
        """获取性能报告"""
        if not self.perf_stats['total_time']:
            return {}
            
        window = min(100, len(self.perf_stats['total_time']))
        report = {
            'avg_total_time': np.mean(self.perf_stats['total_time'][-window:]) * 1000,
            'avg_batch_size': np.mean(self.perf_stats['batch_sizes'][-window:]),
            'memory_usage_mb': np.mean(self.perf_stats['memory_usage'][-window:]) if self.perf_stats['memory_usage'] else 0
        }
        
        return report
    
    def compute_bone_transforms(self, current_pose: List[Dict[str, float]]) -> List[np.ndarray]:
        """
        计算骨骼变换矩阵
        
        Args:
            current_pose: 当前姿态数据
            
        Returns:
            变换矩阵列表
        """
        try:
            if self.device.type == 'cuda':
                return self._compute_bone_transforms_gpu(current_pose)
            else:
                return self._compute_bone_transforms_cpu(current_pose)
            
        except Exception as e:
            logger.error(f"Error computing bone transforms: {str(e)}")
            # 返回单位矩阵作为后备
            return [np.eye(4) for _ in range(len(current_pose))]
    
    def _compute_bone_transforms_gpu(self, current_pose: List[Dict[str, float]]) -> torch.Tensor:
        """
        计算GPU版本的骨骼变换矩阵
        
        Args:
            current_pose: 当前姿态数据
            
        Returns:
            变换矩阵张量
        """
        try:
            batch_size = len(current_pose)
            transforms = torch.eye(4, device=self.device).repeat(batch_size, 1, 1)
            
            for i, pose_data in enumerate(current_pose):
                # 构建变换矩阵
                transform = transforms[i]
                
                # 应用旋转
                if 'rotation' in pose_data:
                    rotation = torch.tensor(pose_data['rotation'], device=self.device)
                    rad_angle = torch.deg2rad(rotation)
                    cos_theta = torch.cos(rad_angle)
                    sin_theta = torch.sin(rad_angle)
                    rotation_matrix = torch.eye(4, device=self.device)
                    rotation_matrix[0:2, 0:2] = torch.tensor([
                        [cos_theta, -sin_theta],
                        [sin_theta, cos_theta]
                    ], device=self.device)
                    transform = torch.mm(transform, rotation_matrix)
                
                # 应用平移
                if 'translation' in pose_data:
                    tx, ty = pose_data['translation']
                    transform[0:2, 3] = torch.tensor([tx, ty], device=self.device)
                
                transforms[i] = transform
            
            return transforms
            
        except Exception as e:
            logger.error("Error computing GPU bone transforms: %s", str(e))
            # 返回单位矩阵作为后备
            return torch.eye(4, device=self.device).repeat(len(current_pose), 1, 1)
    
    def _compute_bone_transforms_cpu(self, current_pose: List[Dict[str, float]]) -> List[np.ndarray]:
        """
        计算CPU版本的骨骼变换矩阵
        
        Args:
            current_pose: 当前姿态数据
            
        Returns:
            变换矩阵列表
        """
        try:
            transforms = []
            for bone_idx, pose_data in enumerate(current_pose):
                # 构建变换矩阵
                transform = np.eye(4)
                
                # 应用旋转
                if 'rotation' in pose_data:
                    rotation = pose_data['rotation']
                    rad_angle = np.radians(rotation)
                    cos_theta = np.cos(rad_angle)
                    sin_theta = np.sin(rad_angle)
                    rotation_matrix = np.array([
                        [cos_theta, -sin_theta, 0, 0],
                        [sin_theta, cos_theta, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]
                    ])
                    transform = np.dot(transform, rotation_matrix)
                
                # 应用平移
                if 'translation' in pose_data:
                    tx, ty = pose_data['translation']
                    transform[0:2, 3] = [tx, ty]
                
                transforms.append(transform)
            
            return transforms
            
        except Exception as e:
            logger.error("Error computing bone transforms: %s", str(e))
            # 返回单位矩阵列表作为后备
            return [np.eye(4) for _ in range(len(current_pose))]
    
    def apply_deformation(self, transforms: Union[List[np.ndarray], torch.Tensor]) -> np.ndarray:
        """
        应用变形
        
        Args:
            transforms: 骨骼变换矩阵列表或张量
            
        Returns:
            变形后的图像
        """
        try:
            if self.device.type == 'cuda':
                return self._apply_deformation_gpu(transforms)
            else:
                return self._apply_deformation_cpu(transforms)
            
        except Exception as e:
            logger.error("Deformation error: %s", str(e))
            return self.binding.reference_frame.copy()
    
    def _apply_deformation_gpu(self, transforms: torch.Tensor) -> np.ndarray:
        """
        GPU版本的变形应用
        
        Args:
            transforms: 骨骼变换矩阵张量
            
        Returns:
            变形后的图像
        """
        try:
            # 将参考帧转换为GPU张量
            reference_frame = torch.from_numpy(self.binding.reference_frame).to(self.device)
            height, width = reference_frame.shape[:2]
            
            # 获取当前批处理大小
            batch_size = self._get_adaptive_batch_size()
            num_points = len(self.binding.mesh_points)
            
            # 准备网格点和权重
            mesh_points = torch.from_numpy(
                np.column_stack((self.binding.mesh_points, np.zeros(num_points), np.ones(num_points)))
            ).float().to(self.device)
            
            weights = torch.from_numpy(self.binding.weights).float().to(self.device)
            
            # 初始化结果
            result = torch.zeros_like(reference_frame)
            
            # 批量处理网格点
            for start_idx in range(0, num_points, batch_size):
                end_idx = min(start_idx + batch_size, num_points)
                batch_points = mesh_points[start_idx:end_idx]
                batch_weights = weights[start_idx:end_idx]
                
                # 计算变形后的点位置
                deformed_points = torch.zeros(
                    (end_idx - start_idx, 2), 
                    device=self.device, 
                    dtype=torch.float32
                )
                
                # 应用加权变换
                for i in range(len(transforms)):
                    weight_mask = batch_weights[:, i] > 0
                    if weight_mask.any():
                        weighted_transform = transforms[i] * batch_weights[weight_mask, i:i+1]
                        deformed_points[weight_mask] += torch.bmm(
                            batch_points[weight_mask].unsqueeze(1),
                            weighted_transform.transpose(0, 1)
                        ).squeeze(1)[:, :2]
                
                # 将变形点映射到图像空间
                deformed_points = torch.clamp(deformed_points, 0, torch.tensor([width-1, height-1], device=self.device))
                
                # 计算局部变换
                if end_idx - start_idx >= 3:
                    src_points = batch_points[:3, :2].cpu().numpy()
                    dst_points = deformed_points[:3].cpu().numpy()
                    
                    transform_matrix = cv2.getAffineTransform(
                        src_points.astype(np.float32),
                        dst_points.astype(np.float32)
                    )
                    
                    transform_tensor = torch.from_numpy(transform_matrix).to(self.device)
                    
                    # 应用变换
                    grid = torch.nn.functional.affine_grid(
                        transform_tensor.unsqueeze(0)[:, :2],
                        reference_frame.unsqueeze(0).shape,
                        align_corners=False
                    )
                    
                    local_result = torch.nn.functional.grid_sample(
                        reference_frame.unsqueeze(0).permute(0, 3, 1, 2),
                        grid,
                        mode='bilinear',
                        padding_mode='reflection',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0)
                    
                    # 创建混合蒙版
                    mask = torch.zeros((height, width), device=self.device)
                    hull_points = deformed_points.cpu().numpy().astype(np.int32)
                    cv2.fillConvexPoly(mask.cpu().numpy(), hull_points, 1)
                    mask = torch.from_numpy(mask).to(self.device)
                    mask = torch.nn.functional.gaussian_blur(
                        mask.unsqueeze(0).unsqueeze(0),
                        kernel_size=[5, 5],
                        sigma=[1.5, 1.5]
                    ).squeeze()
                    
                    # 混合结果
                    result = result * (1 - mask.unsqueeze(-1)) + local_result * mask.unsqueeze(-1)
            
            # 更新性能统计
            self.perf_stats['batch_size'].append(batch_size)
            if self.device.type == 'cuda':
                self.perf_stats['memory_usage'].append(
                    torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                )
            
            return result.cpu().numpy().astype(np.uint8)
            
        except Exception as e:
            logger.error("GPU deformation error: %s", str(e))
            return self.binding.reference_frame.copy()
    
    def _apply_deformation_cpu(self, transforms: List[np.ndarray]) -> np.ndarray:
        """
        CPU版本的变形应用
        
        Args:
            transforms: 骨骼变换矩阵列表
            
        Returns:
            变形后的图像
        """
        try:
            result = np.zeros_like(self.binding.reference_frame)
            height, width = result.shape[:2]
            
            # 获取当前批处理大小
            batch_size = self.batch_size
            num_points = len(self.binding.mesh_points)
            
            # 批量处理网格点
            for start_idx in range(0, num_points, batch_size):
                end_idx = min(start_idx + batch_size, num_points)
                batch_points = self.binding.mesh_points[start_idx:end_idx]
                batch_weights = self.binding.weights[start_idx:end_idx]
                
                # 计算变形后的点位置
                deformed_points = np.zeros_like(batch_points)
                for i, (point, weights) in enumerate(zip(batch_points, batch_weights)):
                    point_homo = np.append(point, [0, 1])  # 齐次坐标
                    transformed_point = np.zeros(4)
                    
                    # 应用加权变换
                    for transform, weight in zip(transforms, weights):
                        if weight > 0:
                            transformed_point += weight * (transform @ point_homo)
                            
                    deformed_points[i] = transformed_point[:2]  # 转回2D坐标
                
                # 将变形点映射到图像空间
                deformed_points = np.clip(deformed_points, 0, [width-1, height-1])
                
                # 应用简单的三角形变形
                if len(deformed_points) >= 3:
                    for i in range(0, len(deformed_points)-2, 3):
                        pts = deformed_points[i:i+3].astype(np.int32)
                        mask = np.zeros((height, width), dtype=np.uint8)
                        cv2.fillConvexPoly(mask, pts, 1)
                        
                        # 计算仿射变换
                        src_pts = batch_points[i:i+3].astype(np.float32)
                        dst_pts = pts.astype(np.float32)
                        
                        transform_matrix = cv2.getAffineTransform(src_pts, dst_pts)
                        warped = cv2.warpAffine(
                            self.binding.reference_frame,
                            transform_matrix,
                            (width, height),
                            borderMode=cv2.BORDER_REFLECT
                        )
                        
                        # 混合结果
                        result = np.where(mask[..., None] > 0, warped, result)
            
            return result
            
        except Exception as e:
            logger.error("CPU deformation error: %s", str(e))
            return self.binding.reference_frame.copy()
    
    def smooth_result(self, result: np.ndarray) -> np.ndarray:
        """
        平滑处理
        
        对变形结果进行时间和空间域的平滑处理。
        
        参数:
            result: 待平滑的图像
            
        返回:
            np.ndarray: 平滑后的图像
        """
        # 添加到帧缓冲
        self.frame_buffer.append(result)
        if len(self.frame_buffer) > 3:
            self.frame_buffer.pop(0)
            
        # 时间域平滑
        if len(self.frame_buffer) > 1:
            result = np.mean(self.frame_buffer, axis=0)
            
        # 空间域平滑
        return cv2.GaussianBlur(result, (3, 3), 
                               self.config.smoothing_factor)
    
    def blend_results(self, result_2d: np.ndarray, 
                     result_3d: np.ndarray) -> np.ndarray:
        """混合2D和3D渲染结果"""
        return cv2.addWeighted(result_2d, self.config.blend_weight,
                             result_3d, 1 - self.config.blend_weight, 0)
    
    def _validate_pose(self, pose_data: List[Dict[str, float]]) -> bool:
        """
        验证姿态数据的有效性
        
        Args:
            pose_data: 姿态数据列表
            
        Returns:
            数据是否有效
        """
        try:
            if not isinstance(pose_data, list):
                return False
            
            if len(pose_data) != len(self.binding.weights[0]):
                return False
            
            for bone_data in pose_data:
                if not isinstance(bone_data, dict):
                    return False
                    
                # 检查必要的变换数据
                if 'rotation' in bone_data:
                    if not isinstance(bone_data['rotation'], (int, float)):
                        return False
                        
                if 'translation' in bone_data:
                    if not isinstance(bone_data['translation'], (list, tuple)):
                        return False
                    if len(bone_data['translation']) != 2:
                        return False
                    if not all(isinstance(x, (int, float)) for x in bone_data['translation']):
                        return False
                        
                if 'scale' in bone_data:
                    if not isinstance(bone_data['scale'], (list, tuple)):
                        return False
                    if len(bone_data['scale']) != 2:
                        return False
                    if not all(isinstance(x, (int, float)) for x in bone_data['scale']):
                        return False
                        
            return True
            
        except Exception as e:
            logger.error("Pose validation error: %s", str(e))
            return False
    
    def _suppress_jitter(self, transforms: List[np.ndarray]) -> List[np.ndarray]:
        """抑制抖动"""
        filtered_transforms = []
        
        for curr, prev in zip(transforms, self.previous_transforms):
            # 计算变换差异
            diff = np.abs(curr - prev)
            
            # 如果差异小于阈值，使用前一帧的变换
            if np.max(diff) < self.config.jitter_threshold:
                filtered_transforms.append(prev)
            else:
                filtered_transforms.append(curr)
                
        return filtered_transforms
    
    def _handle_boundaries(self, result: np.ndarray) -> np.ndarray:
        """处理边界区域"""
        # 添加边界填充
        padded = cv2.copyMakeBorder(result, 
                                  self.config.boundary_padding,
                                  self.config.boundary_padding,
                                  self.config.boundary_padding,
                                  self.config.boundary_padding,
                                  cv2.BORDER_REFLECT)
                                  
        # 填充空洞
        mask = (padded == 0).astype(np.uint8)
        padded = cv2.inpaint(padded, mask, 3, cv2.INPAINT_TELEA)
        
        # 裁剪回原始大小
        return padded[self.config.boundary_padding:-self.config.boundary_padding,
                     self.config.boundary_padding:-self.config.boundary_padding] 
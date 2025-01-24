from typing import List, Dict, Optional
import numpy as np
import cv2
from .pose_data import PoseData, DeformRegion, Landmark
import time
import pytest
from config.settings import POSE_CONFIG


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

    def deform_frame(self,
                     frame: np.ndarray,
                     regions: Dict[str, DeformRegion],
                     target_pose: PoseData) -> np.ndarray:
        """变形图像帧

        Args:
            frame: 输入图像帧
            regions: 区域绑定信息
            target_pose: 目标姿态

        Returns:
            变形后的图像帧
        """
        frame = self._ensure_type_compatibility(frame)
        if frame is None or target_pose is None:
            raise ValueError("Invalid frame or pose data")

        # 添加姿态验证
        if not self._validate_pose(target_pose):
            raise ValueError("Invalid pose data: failed validation checks")

        # 创建输出帧
        result = frame.copy()
        transformed_regions = {}

        # 处理每个区域
        for region_name, region in regions.items():
            # 计算变形矩阵
            transform = self._calculate_transform(region, target_pose)

            # 应用变形到区域
            transformed = self._apply_transform(frame, region, transform)
            transformed_regions[region_name] = transformed

        # 混合所有变形区域
        result = self._blend_regions(frame, transformed_regions)

        # 应用时间平滑 - 确保类型和尺寸匹配
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

    def _apply_transform(self, frame: np.ndarray, region: DeformRegion, transform: np.ndarray) -> np.ndarray:
        """应用变形到区域"""
        height, width = frame.shape[:2]
        result = np.zeros_like(frame)

        # 计算变形区域的边界
        mask = region.mask if region.mask is not None else np.ones((height, width), dtype=np.uint8)
        y, x = np.nonzero(mask)
        if len(x) == 0 or len(y) == 0:
            return result

        # 计算变形区域的边界框
        min_x, max_x = np.min(x), np.max(x)
        min_y, max_y = np.min(y), np.max(y)

        # 扩展边界框以包含过渡区域
        padding = int(self._blend_radius * 2)
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(width, max_x + padding)
        max_y = min(height, max_y + padding)

        # 只变形边界框内的区域
        roi = frame[min_y:max_y, min_x:max_x]
        roi_transform = transform.copy()
        roi_transform[:, 2] -= [min_x, min_y]  # 调整平移分量

        warped_roi = cv2.warpAffine(
            roi,
            roi_transform,
            (max_x - min_x, max_y - min_y),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        # 将变形结果复制回原始位置
        result[min_y:max_y, min_x:max_x] = warped_roi

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

    def batch_deform(self, frame: np.ndarray, poses: List[PoseData]) -> List[np.ndarray]:
        """批量处理多个姿态"""
        regions = {}  # 创建空的regions字典
        results = []
        for pose in poses:
            result = self.deform_frame(frame, regions, pose)
            results.append(result)
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

        # 验证关键点数量
        if len(pose.landmarks) < 33:  # MediaPipe标准关键点数量
            return False

        # 验证坐标有效性
        for lm in pose.landmarks:
            if (np.isnan(lm.x) or np.isnan(lm.y) or np.isnan(lm.z) or
                    np.isinf(lm.x) or np.isinf(lm.y) or np.isinf(lm.z)):
                return False

        # 验证可见度和置信度
        if pose.confidence < 0.5:  # 最小置信度阈值
            return False

        visible_points = sum(1 for lm in pose.landmarks if lm.visibility > 0.5)
        if visible_points < len(pose.landmarks) * 0.6:  # 要求至少60%的点可见
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
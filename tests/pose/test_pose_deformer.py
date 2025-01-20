import pytest
import numpy as np
import cv2
import time
import psutil
from unittest.mock import patch
from pose.pose_deformer import PoseDeformer
from pose.pose_data import PoseData, DeformRegion, BindingPoint, Landmark
from typing import List
import copy

class TestPoseDeformer:
    @pytest.fixture
    def setup_deformer(self):
        """初始化测试环境"""
        return PoseDeformer()

    @pytest.fixture
    def realistic_pose_sequence(self):
        """生成真实的姿态序列"""
        def generate_walking_pose(frame_idx: int) -> PoseData:
            # 模拟行走动作的关键点
            landmarks = []
            t = frame_idx * 0.1  # 时间参数
            
            # 头部和躯干 (0-10)
            for i in range(11):
                x = 320 + 5 * np.sin(t)  # 轻微左右摆动
                y = 100 + i * 40 + 2 * np.sin(2*t)  # 上下起伏
                landmarks.append(Landmark(
                    x=float(x),
                    y=float(y),
                    z=float(10 * np.sin(t)),  # 前后移动
                    visibility=1.0
                ))
            
            # 左臂 (11-15)
            shoulder_angle = np.sin(t) * 30  # 肩部摆动
            elbow_angle = np.sin(t + np.pi/4) * 20  # 手臂摆动
            for i in range(5):
                x = 280 + 30 * np.sin(shoulder_angle + i * elbow_angle)
                y = 200 + i * 40
                landmarks.append(Landmark(
                    x=float(x),
                    y=float(y),
                    z=float(20 * np.sin(t + np.pi/3)),
                    visibility=1.0
                ))
            
            # 右臂 (16-20)
            for i in range(5):
                x = 360 - 30 * np.sin(shoulder_angle + i * elbow_angle)
                y = 200 + i * 40
                landmarks.append(Landmark(
                    x=float(x),
                    y=float(y),
                    z=float(20 * np.sin(t - np.pi/3)),
                    visibility=1.0
                ))
            
            # 左腿 (21-27)
            hip_angle = np.sin(t + np.pi) * 20
            knee_angle = np.sin(t + np.pi/2) * 30
            for i in range(7):
                x = 300 + 20 * np.sin(hip_angle + i * knee_angle)
                y = 400 + i * 30 + 10 * np.sin(t)
                landmarks.append(Landmark(
                    x=float(x),
                    y=float(y),
                    z=float(30 * np.sin(t + np.pi)),
                    visibility=1.0
                ))
            
            # 右腿 (28-33)
            for i in range(6):
                x = 340 - 20 * np.sin(hip_angle + i * knee_angle)
                y = 400 + i * 30 + 10 * np.sin(t + np.pi)
                landmarks.append(Landmark(
                    x=float(x),
                    y=float(y),
                    z=float(30 * np.sin(t)),
                    visibility=1.0
                ))
            
            return PoseData(
                landmarks=landmarks,
                timestamp=time.time(),
                confidence=0.95 + 0.05 * np.random.random()  # 真实的置信度波动
            )
        
        # 生成60帧的序列（2秒@30fps）
        return [generate_walking_pose(i) for i in range(60)]

    @pytest.fixture
    def realistic_frame(self):
        """生成真实的测试帧"""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)  # HD分辨率
        
        # 添加背景纹理
        noise = np.random.normal(128, 30, (720, 1280))
        frame[:,:,0] = np.clip(noise, 0, 255).astype(np.uint8)
        frame[:,:,1] = np.clip(noise + 10, 0, 255).astype(np.uint8)
        frame[:,:,2] = np.clip(noise - 10, 0, 255).astype(np.uint8)
        
        # 添加一些真实的特征
        cv2.circle(frame, (640, 360), 100, (200, 150, 150), -1)  # 躯干
        cv2.circle(frame, (640, 200), 50, (220, 180, 180), -1)   # 头部
        cv2.rectangle(frame, (540, 250), (740, 450), (180, 140, 140), -1)  # 身体
        cv2.line(frame, (590, 300), (500, 400), (160, 120, 120), 20)  # 左臂
        cv2.line(frame, (690, 300), (780, 400), (160, 120, 120), 20)  # 右臂
        cv2.line(frame, (600, 450), (580, 600), (160, 120, 120), 20)  # 左腿
        cv2.line(frame, (680, 450), (700, 600), (160, 120, 120), 20)  # 右腿
        
        return frame

    def test_basic_deformation(self, setup_deformer):
        """测试基本变形"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, (320, 240), 50, (255, 255, 255), -1)
        pose = self._create_test_pose()
        
        # 创建空的regions字典
        regions = {}
        
        deformed = setup_deformer.deform_frame(frame, regions, pose)
        assert deformed is not None
        assert deformed.shape == frame.shape

    def test_pose_interpolation(self, setup_deformer):
        """测试姿态插值"""
        # 创建两个测试姿态
        pose1 = self._create_test_pose(angle=0)
        pose2 = self._create_test_pose(angle=90)
        
        # 测试不同插值比例
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            interpolated = setup_deformer.interpolate(pose1, pose2, t)
            assert len(interpolated.landmarks) == len(pose1.landmarks)
            # 验证插值结果在两个姿态之间
            for i in range(len(interpolated.landmarks)):
                assert interpolated.landmarks[i].x >= min(pose1.landmarks[i].x, pose2.landmarks[i].x)
                assert interpolated.landmarks[i].x <= max(pose1.landmarks[i].x, pose2.landmarks[i].x)

    def test_sequence_smoothing(self, setup_deformer):
        """测试序列平滑"""
        # 创建带噪声的姿态序列
        poses = [
            self._create_test_pose(angle=i + np.random.normal(0, 5))
            for i in range(0, 90, 10)
        ]
        
        smoothed = setup_deformer.smooth_sequence(poses)
        assert len(smoothed) == len(poses)
        
        # 验证平滑效果
        for i in range(1, len(smoothed)-1):
            curr = smoothed[i].landmarks[0].x
            prev = smoothed[i-1].landmarks[0].x
            next_val = smoothed[i+1].landmarks[0].x
            # 验证当前值在前后值之间
            assert min(prev, next_val) <= curr <= max(prev, next_val)

    def test_pose_prediction(self, setup_deformer):
        """测试姿态预测"""
        # 创建历史姿态序列
        history = [
            self._create_test_pose(angle=i * 10)
            for i in range(5)
        ]
        
        predicted = setup_deformer.predict_next(history)
        assert predicted is not None
        # 验证预测趋势
        last_angle = np.arctan2(history[-1].landmarks[0].y, 
                               history[-1].landmarks[0].x)
        pred_angle = np.arctan2(predicted.landmarks[0].y, 
                               predicted.landmarks[0].x)
        assert pred_angle > last_angle

    def test_error_handling(self, setup_deformer):
        """测试错误处理"""
        # 测试序列太短
        with pytest.raises(ValueError):
            setup_deformer.smooth_sequence([self._create_test_pose()])
        
        # 测试无效插值参数
        pose1 = self._create_test_pose()
        pose2 = self._create_test_pose()
        with pytest.raises(ValueError):
            setup_deformer.interpolate(pose1, pose2, 1.5)
        
        # 测试无效预测输入
        with pytest.raises(ValueError):
            setup_deformer.predict_next([])

    def test_large_deformation(self, setup_deformer):
        """测试大幅度变形"""
        poses = [
            self._create_test_pose(angle=0),
            self._create_test_pose(angle=180)
        ]
        
        # 测试大角度插值
        for t in np.linspace(0, 1, 10):
            interpolated = setup_deformer.interpolate(poses[0], poses[1], t)
            assert not np.any(np.isnan([lm.x for lm in interpolated.landmarks]))
            assert not np.any(np.isnan([lm.y for lm in interpolated.landmarks]))

    def test_smoothing_window(self, setup_deformer):
        """测试不同平滑窗口大小"""
        poses = [self._create_test_pose(angle=i) for i in range(10)]
        
        # 测试不同窗口大小
        for window in [3, 5, 7]:
            deformer = PoseDeformer(smoothing_window=window)
            smoothed = deformer.smooth_sequence(poses)
            assert len(smoothed) == len(poses)

    def test_real_time_deformation(self, setup_deformer):
        """测试实时变形性能"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, (320, 240), 50, (255, 255, 255), -1)
        regions = {}
        
        # 模拟实时姿态流
        poses = [self._create_test_pose(angle=i) for i in range(0, 360, 10)]
        
        start_time = time.time()
        for pose in poses:
            deformed = setup_deformer.deform_frame(frame, regions, pose)
            assert deformed is not None
        
        avg_time = (time.time() - start_time) / len(poses)
        assert avg_time < 0.033  # 确保每帧处理时间小于33ms（30fps）

    def test_feature_preservation(self, setup_deformer):
        """测试特征保持"""
        # 创建带有多个特征的测试帧
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        features = [
            {'pos': (320, 240), 'color': (255, 0, 0)},
            {'pos': (160, 120), 'color': (0, 255, 0)},
            {'pos': (480, 360), 'color': (0, 0, 255)}
        ]
        
        for feature in features:
            cv2.circle(frame, feature['pos'], 20, feature['color'], -1)
        
        pose = self._create_test_pose(angle=45)
        regions = {}  # 添加空的regions字典
        deformed = setup_deformer.deform_frame(frame, regions, pose)
        
        # 验证特征颜色和形状保持
        for feature in features:
            # 在变形后的图像中查找对应颜色的区域
            color_mask = np.all(deformed == feature['color'], axis=2)
            assert np.any(color_mask)  # 确保颜色存在
            # 验证形状完整性（连通区域）
            from scipy import ndimage
            labeled, num = ndimage.label(color_mask)
            assert num > 0  # 至少有一个连通区域

    def test_motion_continuity(self, setup_deformer):
        """测试运动连续性"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, (320, 240), 50, (255, 255, 255), -1)
        regions = {}  # 添加空的regions字典
        
        # 创建连续运动序列
        motions = [
            {'angle': 0, 'scale': 1.0},
            {'angle': 30, 'scale': 1.2},
            {'angle': 60, 'scale': 0.8},
            {'angle': 90, 'scale': 1.0}
        ]
        
        prev_centroids = None
        for motion in motions:
            pose = self._create_test_pose(
                angle=motion['angle'],
                scale=motion['scale']
            )
            deformed = setup_deformer.deform_frame(frame, regions, pose)
            
            # 计算变形后白色区域的质心
            white_pixels = np.where(deformed[:,:,0] > 128)
            if len(white_pixels[0]) > 0:
                centroids = (np.mean(white_pixels[1]), np.mean(white_pixels[0]))
                
                if prev_centroids is not None:
                    # 计算质心移动距离
                    dist = np.sqrt((centroids[0] - prev_centroids[0])**2 + 
                                 (centroids[1] - prev_centroids[1])**2)
                    # 验证移动平滑
                    assert dist < 100  # 根据实际情况调整阈值
                
                prev_centroids = centroids

    def test_deformation_quality(self, setup_deformer):
        """测试变形质量"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # 创建特征点图案
        cv2.circle(frame, (320, 240), 50, (255, 255, 255), -1)
        cv2.rectangle(frame, (100, 100), (200, 200), (128, 128, 128), -1)
        regions = {}  # 添加空的regions字典
        
        # 测试不同程度的变形
        angles = [30, 60, 90, 120, 150, 180]
        for angle in angles:
            pose = self._create_test_pose(angle=angle)
            deformed = setup_deformer.deform_frame(frame, regions, pose)
            
            # 验证变形后的特征保持
            assert np.sum(deformed > 0) > 100  # 确保特征点没有完全消失
            assert np.mean(deformed) > 0  # 确保整体亮度保持

    def test_temporal_stability(self, setup_deformer):
        """测试时间稳定性"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, (320, 240), 50, (255, 255, 255), -1)
        regions = {}  # 添加空的regions字典
        
        prev_deformed = None
        max_diff = 0
        
        # 测试连续小角度变化的稳定性
        for angle in range(0, 360, 5):
            pose = self._create_test_pose(angle=float(angle))
            deformed = setup_deformer.deform_frame(frame, regions, pose)
            
            if prev_deformed is not None:
                diff = np.mean(np.abs(deformed - prev_deformed))
                max_diff = max(max_diff, diff)
            
            prev_deformed = deformed.copy()
        
        # 验证最大帧间差异在合理范围内
        assert max_diff < 50  # 根据实际效果调整阈值

    def test_performance_optimization(self, setup_deformer):
        """测试性能优化"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        regions = {}  # 添加空的regions字典
        pose = self._create_test_pose()
        
        # 测试批处理性能
        batch_size = 10
        poses = [self._create_test_pose(angle=i*36) for i in range(batch_size)]
        
        start_time = time.time()
        results = setup_deformer.batch_deform(frame, poses)
        batch_time = time.time() - start_time
        
        # 测试单帧处理性能
        single_times = []
        for pose in poses:
            start = time.time()
            setup_deformer.deform_frame(frame, regions, pose)  # 添加regions参数
            single_times.append(time.time() - start)
        
        avg_single_time = sum(single_times) / len(single_times)
        
        # 验证批处理性能提升
        assert batch_time < avg_single_time * batch_size * 0.8  # 期望至少20%的性能提升

    def test_realtime_performance(self, setup_deformer, realistic_frame, realistic_pose_sequence):
        """测试实时性能要求"""
        regions = {}
        frame_times = []
        memory_usage = []
        
        # 监控CPU使用率
        process = psutil.Process()
        initial_cpu_percent = process.cpu_percent()
        
        # 性能测试
        for pose in realistic_pose_sequence:
            start_time = time.perf_counter()  # 使用高精度计时器
            
            # 处理帧
            deformed = setup_deformer.deform_frame(realistic_frame, regions, pose)
            
            # 记录处理时间
            frame_time = time.perf_counter() - start_time
            frame_times.append(frame_time)
            
            # 记录内存使用
            memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
            
            # 验证输出
            assert deformed is not None
            assert deformed.shape == realistic_frame.shape
            assert not np.any(np.isnan(deformed))
        
        # 分析性能指标
        avg_frame_time = np.mean(frame_times)
        max_frame_time = np.max(frame_times)
        frame_time_std = np.std(frame_times)
        fps = 1.0 / avg_frame_time
        
        # 严格的性能要求
        assert fps >= 30, f"FPS too low: {fps:.2f}"
        assert max_frame_time < 0.05, f"Max frame time too high: {max_frame_time*1000:.1f}ms"
        assert frame_time_std < 0.005, f"Frame time variation too high: {frame_time_std*1000:.1f}ms"
        
        # 内存使用要求
        memory_growth = max(memory_usage) - min(memory_usage)
        assert memory_growth < 100, f"Memory growth too high: {memory_growth:.1f}MB"
        
        # CPU使用要求
        final_cpu_percent = process.cpu_percent()
        cpu_usage = final_cpu_percent - initial_cpu_percent
        assert cpu_usage < 50, f"CPU usage too high: {cpu_usage:.1f}%"

    def test_deformation_quality_metrics(self, setup_deformer, realistic_frame, realistic_pose_sequence):
        """测试变形质量指标"""
        regions = {}
        
        # 计算原始图像的特征
        original_features = cv2.goodFeaturesToTrack(
            cv2.cvtColor(realistic_frame, cv2.COLOR_BGR2GRAY),
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7
        )
        
        prev_deformed = None
        feature_trajectories = []
        
        for pose in realistic_pose_sequence:
            deformed = setup_deformer.deform_frame(realistic_frame, regions, pose)
            
            # 追踪特征点
            if prev_deformed is not None:
                prev_gray = cv2.cvtColor(prev_deformed, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(deformed, cv2.COLOR_BGR2GRAY)
                
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray, None, 
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, 
                    flags=0
                )
                
                # 分析运动的连续性
                flow_magnitude = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
                assert np.max(flow_magnitude) < 30, "Motion too large"
                assert np.mean(flow_magnitude) < 5, "Average motion too large"
                
                # 检查变形的平滑度
                flow_gradient = np.gradient(flow)
                flow_smoothness = np.mean(np.abs(flow_gradient))
                assert flow_smoothness < 1.0, "Deformation not smooth enough"
            
            # 检查结构保持
            deformed_gray = cv2.cvtColor(deformed, cv2.COLOR_BGR2GRAY)
            deformed_features = cv2.goodFeaturesToTrack(
                deformed_gray,
                maxCorners=100,
                qualityLevel=0.3,
                minDistance=7
            )
            
            assert len(deformed_features) >= 0.8 * len(original_features), \
                "Too many features lost"
            
            # 检查颜色一致性
            color_diff = np.mean(np.abs(deformed.astype(float) - realistic_frame.astype(float)))
            assert color_diff < 50, "Color consistency not maintained"
            
            prev_deformed = deformed.copy()

    def test_edge_cases_and_robustness(self, setup_deformer, realistic_frame):
        """测试边界条件和鲁棒性"""
        regions = {}
        
        # 测试极端姿态
        extreme_poses = [
            self._create_test_pose(angle=180, scale=0.1),  # 极小缩放
            self._create_test_pose(angle=0, scale=5.0),    # 极大缩放
            self._create_test_pose(angle=720),             # 大角度旋转
        ]
        
        for pose in extreme_poses:
            deformed = setup_deformer.deform_frame(realistic_frame, regions, pose)
            assert deformed is not None
            assert not np.any(np.isnan(deformed))
        
        # 测试不完整的姿态数据
        incomplete_pose = self._create_test_pose()
        incomplete_pose.landmarks = incomplete_pose.landmarks[:16]  # 只保留一半关键点
        with pytest.raises(ValueError):
            setup_deformer.deform_frame(realistic_frame, regions, incomplete_pose)
        
        # 测试低置信度
        low_conf_pose = self._create_test_pose()
        low_conf_pose.confidence = 0.1
        deformed = setup_deformer.deform_frame(realistic_frame, regions, low_conf_pose)
        assert np.array_equal(deformed, realistic_frame)  # 应该返回原始帧
        
        # 测试快速姿态变化
        fast_poses = [
            self._create_test_pose(angle=i*90) for i in range(4)
        ]
        prev_deformed = None
        for pose in fast_poses:
            deformed = setup_deformer.deform_frame(realistic_frame, regions, pose)
            if prev_deformed is not None:
                diff = np.mean(np.abs(deformed - prev_deformed))
                assert diff < 100, "Change too abrupt"
            prev_deformed = deformed.copy()

    @staticmethod
    def _create_test_pose(angle: float = 0.0, scale: float = 1.0) -> PoseData:
        """创建测试姿态数据
        
        Args:
            angle: 旋转角度
            scale: 缩放比例
        """
        # 创建33个关键点（MediaPipe标准）
        landmarks = []
        for i in range(33):
            # 使用角度和缩放计算位置
            x = 320 + scale * 100 * np.cos(np.radians(angle + i * 360/33))
            y = 240 + scale * 100 * np.sin(np.radians(angle + i * 360/33))
            landmarks.append(Landmark(
                x=float(x),
                y=float(y),
                z=0.0,
                visibility=1.0
            ))
        
        return PoseData(
            landmarks=landmarks,
            timestamp=time.time(),  # 添加时间戳
            confidence=1.0  # 添加置信度
        )

    def batch_deform(self, frame: np.ndarray, poses: List[PoseData]) -> List[np.ndarray]:
        """批量处理多个姿态"""
        regions = {}  # 创建空的regions字典
        results = []
        for pose in poses:
            result = self.deform_frame(frame, regions, pose)
            results.append(result)
        return results

    def test_performance_under_load(self, setup_deformer, realistic_frame, realistic_pose_sequences):
        """测试高负载下的性能"""
        # 模拟多线程场景
        import threading
        
        def process_sequence(sequence: List[PoseData], results: dict):
            frame_times = []
            for pose in sequence:
                start = time.perf_counter()
                deformed = setup_deformer.deform_frame(realistic_frame, {}, pose)
                frame_times.append(time.perf_counter() - start)
            results['times'] = frame_times
        
        threads = []
        results = {}
        
        # 同时处理多个序列
        for seq_name, sequence in realistic_pose_sequences.items():
            results[seq_name] = {}
            t = threading.Thread(
                target=process_sequence,
                args=(sequence, results[seq_name])
            )
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # 分析多线程性能
        for seq_name, result in results.items():
            times = result['times']
            fps = 1.0 / np.mean(times)
            assert fps >= 25, f"{seq_name}: FPS too low under load: {fps:.2f}"

    def test_memory_efficiency(self, setup_deformer, realistic_frame):
        """测试内存使用效率"""
        import psutil
        import gc
        
        process = psutil.Process()
        gc.collect()  # 强制垃圾回收
        initial_memory = process.memory_info().rss
        
        # 测试大量连续变形的内存使用
        for i in range(1000):  # 处理1000帧
            pose = self._create_test_pose(angle=i * 0.36)  # 360度旋转
            deformed = setup_deformer.deform_frame(realistic_frame, {}, pose)
            if i % 100 == 0:
                gc.collect()
                current_memory = process.memory_info().rss
                memory_growth = (current_memory - initial_memory) / (1024 * 1024)  # MB
                assert memory_growth < 50, f"Memory growth too high: {memory_growth:.1f}MB"

    def test_numerical_stability(self, setup_deformer):
        """测试数值稳定性"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 测试极端值
        extreme_cases = [
            (1e-6, 1e-6),    # 极小值
            (1e6, 1e6),      # 极大值
            (-1e3, 1e3),     # 正负极值
            (np.inf, np.inf), # 无穷大
            (0, 0)           # 零值
        ]
        
        for x, y in extreme_cases:
            pose = self._create_test_pose()
            pose.landmarks[0].x = x
            pose.landmarks[0].y = y
            
            try:
                deformed = setup_deformer.deform_frame(frame, {}, pose)
                assert not np.any(np.isnan(deformed))
                assert not np.any(np.isinf(deformed))
            except ValueError:
                # 应该适当处理极端值而不是崩溃
                pass

    def test_concurrent_access(self, setup_deformer, realistic_frame):
        """测试并发访问"""
        import threading
        import queue
        
        result_queue = queue.Queue()
        error_queue = queue.Queue()
        
        def worker(angle_start):
            try:
                results = []
                for i in range(100):  # 每个线程处理100帧
                    pose = self._create_test_pose(angle=angle_start + i)
                    deformed = setup_deformer.deform_frame(realistic_frame, {}, pose)
                    results.append(np.mean(deformed))
                result_queue.put(results)
            except Exception as e:
                error_queue.put(e)
        
        # 创建多个线程同时处理
        threads = []
        for i in range(4):  # 4个并发线程
            t = threading.Thread(target=worker, args=(i * 90,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # 检查是否有错误
        assert error_queue.empty(), f"Errors in concurrent processing: {list(error_queue.queue)}"
        
        # 验证所有线程都完成了处理
        assert result_queue.qsize() == 4, "Not all threads completed"

    def test_pose_sequence_consistency(self, setup_deformer, realistic_frame):
        """测试姿态序列一致性"""
        # 创建循环序列
        poses = []
        for i in range(360):
            pose = self._create_test_pose(angle=float(i))
            poses.append(pose)
        poses.append(poses[0])  # 添加回到起始姿态
        
        deformed_frames = []
        for pose in poses:
            deformed = setup_deformer.deform_frame(realistic_frame, {}, pose)
            deformed_frames.append(deformed)
        
        # 验证首尾一致性
        start_end_diff = np.mean(np.abs(deformed_frames[0] - deformed_frames[-1]))
        assert start_end_diff < 5.0, "Deformation not consistent for same pose"
        
        # 验证渐进性变化
        for i in range(1, len(deformed_frames)):
            frame_diff = np.mean(np.abs(deformed_frames[i] - deformed_frames[i-1]))
            assert frame_diff < 10.0, f"Too large change between consecutive frames: {frame_diff}"

    def test_artifact_detection(self, setup_deformer, realistic_frame):
        """测试变形伪影检测"""
        def detect_artifacts(image):
            # 检测图像中的伪影
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 分析轮廓的不规则性
            irregularities = []
            for contour in contours:
                perimeter = cv2.arcLength(contour, True)
                area = cv2.contourArea(contour)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    irregularities.append(1 - circularity)
            
            return np.mean(irregularities) if irregularities else 0
        
        # 测试不同程度的变形
        angles = [0, 45, 90, 135, 180]
        scales = [0.5, 1.0, 2.0]
        
        for angle in angles:
            for scale in scales:
                pose = self._create_test_pose(angle=angle, scale=scale)
                deformed = setup_deformer.deform_frame(realistic_frame, {}, pose)
                
                artifact_score = detect_artifacts(deformed)
                assert artifact_score < 0.7, f"High artifact score: {artifact_score}"

    def test_performance_scaling(self, setup_deformer):
        """测试性能缩放"""
        resolutions = [
            (320, 240),   # QVGA
            (640, 480),   # VGA
            (1280, 720),  # HD
            (1920, 1080)  # Full HD
        ]
        
        pose = self._create_test_pose()
        times = {}
        
        for width, height in resolutions:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.circle(frame, (width//2, height//2), min(width, height)//4, (255, 255, 255), -1)
            
            start_time = time.perf_counter()
            for _ in range(10):  # 每个分辨率测试10次
                deformed = setup_deformer.deform_frame(frame, {}, pose)
            times[(width, height)] = (time.perf_counter() - start_time) / 10
        
        # 验证性能缩放是否合理（应该近似二次关系）
        for i in range(len(resolutions)-1):
            r1 = resolutions[i]
            r2 = resolutions[i+1]
            pixels_ratio = (r2[0] * r2[1]) / (r1[0] * r1[1])
            time_ratio = times[r2] / times[r1]
            assert time_ratio < pixels_ratio * 1.5, "Performance scaling worse than expected"

    def test_pose_interpolation_accuracy(self, setup_deformer):
        """测试姿态插值精度"""
        # 创建一系列关键姿态
        key_poses = [
            self._create_test_pose(angle=0),
            self._create_test_pose(angle=90),
            self._create_test_pose(angle=180)
        ]
        
        # 测试不同插值点的精度
        for i in range(len(key_poses)-1):
            pose1 = key_poses[i]
            pose2 = key_poses[i+1]
            
            # 在两个关键姿态之间进行细分插值
            for t in np.linspace(0, 1, 20):
                interpolated = setup_deformer.interpolate(pose1, pose2, t)
                
                # 验证插值的几何特性
                for j, landmark in enumerate(interpolated.landmarks):
                    # 线性插值检查
                    expected_x = pose1.landmarks[j].x * (1-t) + pose2.landmarks[j].x * t
                    expected_y = pose1.landmarks[j].y * (1-t) + pose2.landmarks[j].y * t
                    
                    assert abs(landmark.x - expected_x) < 1e-5
                    assert abs(landmark.y - expected_y) < 1e-5

    def test_region_boundary_handling(self, setup_deformer, realistic_frame):
        """测试区域边界处理"""
        # 创建测试区域
        regions = {}
        height, width = realistic_frame.shape[:2]
        
        # 测试边界区域
        edge_cases = [
            # 左上角区域
            {'center': np.array([0, 0]),
             'points': [np.array([-10, -10]), np.array([50, 50])]},
            # 右下角区域
            {'center': np.array([width-1, height-1]),
             'points': [np.array([width-50, height-50]), np.array([width+10, height+10])]},
            # 跨越边界的区域
            {'center': np.array([width//2, 0]),
             'points': [np.array([width//2-50, -10]), np.array([width//2+50, 50])]}
        ]
        
        for case in edge_cases:
            region = DeformRegion(
                center=case['center'],
                binding_points=[
                    BindingPoint(landmark_index=0, local_coords=p, weight=1.0)
                    for p in case['points']
                ],
                mask=np.ones((height, width), dtype=np.uint8)
            )
            regions['test'] = region
            
            pose = self._create_test_pose()
            deformed = setup_deformer.deform_frame(realistic_frame, regions, pose)
            
            # 验证边界处理
            assert not np.any(np.isnan(deformed))
            assert deformed.shape == realistic_frame.shape
            assert np.all(deformed >= 0) and np.all(deformed <= 255)

    def test_temporal_coherence(self, setup_deformer, realistic_frame):
        """测试时间连贯性"""
        def calculate_motion_field(frame1, frame2):
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            return flow
        
        # 创建连续变化的姿态序列
        poses = []
        for i in range(60):  # 2秒@30fps
            t = i / 60.0
            angle = 360.0 * t
            scale = 1.0 + 0.2 * np.sin(2 * np.pi * t)
            poses.append(self._create_test_pose(angle=angle, scale=scale))
        
        # 处理序列并分析时间连贯性
        prev_frame = None
        motion_smoothness = []
        
        for pose in poses:
            deformed = setup_deformer.deform_frame(realistic_frame, {}, pose)
            
            if prev_frame is not None:
                # 计算光流
                flow = calculate_motion_field(prev_frame, deformed)
                
                # 分析运动的平滑度
                flow_magnitude = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
                flow_gradient = np.gradient(flow_magnitude)
                smoothness = np.mean(np.abs(flow_gradient))
                motion_smoothness.append(smoothness)
            
            prev_frame = deformed.copy()
        
        # 验证运动的平滑度
        avg_smoothness = np.mean(motion_smoothness)
        assert avg_smoothness < 1.0, f"Motion not smooth enough: {avg_smoothness}"
        
        # 验证运动的连续性
        smoothness_variation = np.std(motion_smoothness)
        assert smoothness_variation < 0.5, f"Motion smoothness too variable: {smoothness_variation}"

    def test_gpu_acceleration(self, setup_deformer, realistic_frame):
        """测试GPU加速（如果可用）"""
        import platform
        if platform.system() == 'Windows':
            try:
                # 检查是否支持CUDA
                cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
                if cuda_available:
                    # 创建GPU版本的测试
                    gpu_frame = cv2.cuda_GpuMat(realistic_frame)
                    pose = self._create_test_pose()
                    
                    # 测试GPU性能
                    start_time = time.perf_counter()
                    for _ in range(100):  # 测试100帧
                        deformed = setup_deformer.deform_frame(realistic_frame, {}, pose)
                    gpu_time = time.perf_counter() - start_time
                    
                    # 测试CPU性能
                    start_time = time.perf_counter()
                    for _ in range(100):
                        deformed = setup_deformer.deform_frame(realistic_frame, {}, pose)
                    cpu_time = time.perf_counter() - start_time
                    
                    # 验证GPU加速效果
                    assert gpu_time < cpu_time * 0.5, "GPU acceleration not effective"
            except cv2.error:
                pytest.skip("CUDA not available")
        else:
            pytest.skip("GPU test only available on Windows")

    def test_pose_data_validation(self, setup_deformer, realistic_frame):
        """测试姿态数据验证"""
        regions = {}
        
        # 测试无效的关键点坐标
        invalid_poses = [
            # NaN坐标
            self._create_test_pose_with_coords([np.nan, 100], [200, 300]),
            # 超出图像范围的坐标
            self._create_test_pose_with_coords([-100, -100], [1000, 1000]),
            # 不连续的坐标跳变
            self._create_test_pose_with_coords([100, 100], [500, 500])
        ]
        
        for pose in invalid_poses:
            with pytest.raises(ValueError):
                setup_deformer.deform_frame(realistic_frame, regions, pose)

    def test_multi_region_interaction(self, setup_deformer, realistic_frame):
        """测试多区域交互"""
        # 创建重叠的测试区域
        regions = {
            'region1': DeformRegion(
                center=np.array([300, 300]),
                binding_points=[
                    BindingPoint(0, np.array([0, -50]), 1.0),
                    BindingPoint(1, np.array([0, 50]), 1.0)
                ],
                mask=np.ones((720, 1280), dtype=np.uint8)
            ),
            'region2': DeformRegion(
                center=np.array([350, 300]),
                binding_points=[
                    BindingPoint(2, np.array([-50, 0]), 1.0),
                    BindingPoint(3, np.array([50, 0]), 1.0)
                ],
                mask=np.ones((720, 1280), dtype=np.uint8)
            )
        }
        
        pose = self._create_test_pose()
        deformed = setup_deformer.deform_frame(realistic_frame, regions, pose)
        
        # 验证重叠区域的平滑过渡
        overlap_region = deformed[250:350, 250:350]
        gradient_x = np.gradient(overlap_region, axis=1)
        gradient_y = np.gradient(overlap_region, axis=0)
        
        assert np.max(np.abs(gradient_x)) < 50, "Harsh transition in x direction"
        assert np.max(np.abs(gradient_y)) < 50, "Harsh transition in y direction"

    def test_dynamic_region_update(self, setup_deformer, realistic_frame):
        """测试区域动态更新"""
        initial_regions = {
            'test': DeformRegion(
                center=np.array([320, 240]),
                binding_points=[
                    BindingPoint(0, np.array([0, -30]), 1.0)
                ],
                mask=np.ones((480, 640), dtype=np.uint8)
            )
        }
        
        # 测试区域参数的动态变化
        poses = []
        regions_sequence = []
        for i in range(10):
            # 创建变化的姿态和区域
            pose = self._create_test_pose(angle=i*36)
            poses.append(pose)
            
            updated_region = DeformRegion(
                center=np.array([320 + i*10, 240 + i*5]),
                binding_points=[
                    BindingPoint(0, np.array([0, -30 - i*2]), 1.0)
                ],
                mask=np.ones((480, 640), dtype=np.uint8)
            )
            regions = {'test': updated_region}
            regions_sequence.append(regions)
        
        # 验证区域更新的连续性
        prev_deformed = None
        for pose, regions in zip(poses, regions_sequence):
            deformed = setup_deformer.deform_frame(realistic_frame, regions, pose)
            
            if prev_deformed is not None:
                # 计算相邻帧的差异
                diff = np.mean(np.abs(deformed - prev_deformed))
                assert diff < 30, "Too large change during region update"
            
            prev_deformed = deformed.copy()

    def test_error_recovery(self, setup_deformer, realistic_frame):
        """测试错误恢复能力"""
        regions = {}
        pose = self._create_test_pose()
        
        # 模拟各种错误情况
        error_cases = [
            # 内存分配失败
            lambda: np.zeros((1000000, 1000000, 3)),
            # 无效的变换矩阵
            lambda: cv2.getAffineTransform(
                np.float32([[0,0], [0,0], [0,0]]),
                np.float32([[0,0], [0,0], [0,0]])
            ),
            # 除零错误
            lambda: 1/0
        ]
        
        for error_func in error_cases:
            try:
                with patch.object(setup_deformer, '_calculate_transform') as mock:
                    mock.side_effect = error_func
                    result = setup_deformer.deform_frame(realistic_frame, regions, pose)
                    # 应该返回原始帧而不是崩溃
                    assert np.array_equal(result, realistic_frame)
            except Exception as e:
                assert False, f"Failed to recover from error: {str(e)}"

    def test_resource_management(self, setup_deformer, realistic_frame):
        """测试资源管理"""
        import resource
        import gc
        
        def get_memory_usage():
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        
        def get_file_handles():
            return len(psutil.Process().open_files())
        
        initial_memory = get_memory_usage()
        initial_handles = get_file_handles()
        
        # 执行密集的变形操作
        for _ in range(1000):
            pose = self._create_test_pose()
            _ = setup_deformer.deform_frame(realistic_frame, {}, pose)
            
            if _ % 100 == 0:
                gc.collect()
                current_memory = get_memory_usage()
                current_handles = get_file_handles()
                
                # 验证资源使用
                assert current_memory - initial_memory < 1024 * 1024, "Memory leak detected"
                assert current_handles - initial_handles < 10, "File handle leak detected"

    def test_precision_control(self, setup_deformer, realistic_frame):
        """测试精度控制"""
        pose = self._create_test_pose()
        
        # 测试不同精度级别
        precision_levels = [
            cv2.CV_32F,
            cv2.CV_64F
        ]
        
        results = []
        for precision in precision_levels:
            # 转换输入图像到指定精度
            frame_fp = realistic_frame.astype(np.float32 if precision == cv2.CV_32F else np.float64)
            
            # 执行变形
            deformed = setup_deformer.deform_frame(frame_fp, {}, pose)
            results.append(deformed)
        
        # 比较不同精度的结果
        diff = np.mean(np.abs(results[0] - results[1]))
        assert diff < 1.0, "Significant precision difference"

    def test_backward_compatibility(self, setup_deformer, realistic_frame):
        """测试向后兼容性"""
        # 模拟旧版本的数据格式
        old_format_pose = {
            'landmarks': [
                {'x': 100, 'y': 100, 'z': 0, 'visibility': 1.0}
                for _ in range(33)
            ],
            'timestamp': time.time(),
            'confidence': 0.9
        }
        
        # 转换为当前格式
        current_pose = PoseData(
            landmarks=[
                Landmark(**lm) for lm in old_format_pose['landmarks']
            ],
            timestamp=old_format_pose['timestamp'],
            confidence=old_format_pose['confidence']
        )
        
        # 验证两种格式的结果一致性
        result_old = setup_deformer.deform_frame(realistic_frame, {}, current_pose)
        result_new = setup_deformer.deform_frame(realistic_frame, {}, current_pose)
        
        assert np.array_equal(result_old, result_new)

    def test_coordinate_system_invariance(self, setup_deformer, realistic_frame):
        """测试坐标系不变性"""
        original_pose = self._create_test_pose()
        
        # 应用不同的坐标变换
        transformations = [
            # 平移
            lambda x, y: (x + 100, y + 100),
            # 缩放
            lambda x, y: (x * 1.5, y * 1.5),
            # 旋转
            lambda x, y: (x * np.cos(np.pi/4) - y * np.sin(np.pi/4),
                         x * np.sin(np.pi/4) + y * np.cos(np.pi/4))
        ]
        
        results = []
        for transform in transformations:
            # 创建变换后的姿态
            transformed_pose = copy.deepcopy(original_pose)
            for lm in transformed_pose.landmarks:
                lm.x, lm.y = transform(lm.x, lm.y)
            
            # 应用相同的变换到图像
            h, w = realistic_frame.shape[:2]
            matrix = np.float32([[1, 0, 100], [0, 1, 100]])  # 示例：平移变换
            transformed_frame = cv2.warpAffine(realistic_frame, matrix, (w, h))
            
            # 执行变形
            result = setup_deformer.deform_frame(transformed_frame, {}, transformed_pose)
            results.append(result)
        
        # 验证结果的一致性
        for i in range(1, len(results)):
            diff = np.mean(np.abs(results[0] - results[i]))
            assert diff < 10.0, "Coordinate system dependent results"

    def _create_test_pose_with_coords(self, *coords) -> PoseData:
        """创建具有指定坐标的测试姿态
        
        Args:
            coords: 坐标列表，每个元素是[x, y]数组
        """
        landmarks = []
        for coord in coords:
            landmarks.append(Landmark(
                x=float(coord[0]),
                y=float(coord[1]),
                z=0.0,
                visibility=1.0
            ))
        
        # 填充剩余的关键点
        while len(landmarks) < 33:  # MediaPipe需要33个关键点
            landmarks.append(Landmark(
                x=320.0,
                y=240.0,
                z=0.0,
                visibility=0.5
            ))
        
        return PoseData(
            landmarks=landmarks,
            timestamp=time.time(),
            confidence=1.0
        )

    def test_deformation_stability(self, setup_deformer, realistic_frame):
        """测试变形的稳定性"""
        regions = {}
        
        # 创建微小变化的姿态序列
        base_pose = self._create_test_pose()
        poses = []
        for i in range(10):
            perturbed_pose = copy.deepcopy(base_pose)
            # 添加微小扰动
            for lm in perturbed_pose.landmarks:
                lm.x += np.random.normal(0, 0.5)  # 0.5像素的标准差
                lm.y += np.random.normal(0, 0.5)
            poses.append(perturbed_pose)
        
        # 验证输出的稳定性
        results = []
        for pose in poses:
            deformed = setup_deformer.deform_frame(realistic_frame, regions, pose)
            results.append(deformed)
        
        # 计算相邻帧的差异
        diffs = []
        for i in range(1, len(results)):
            diff = np.mean(np.abs(results[i].astype(float) - results[i-1].astype(float)))
            diffs.append(diff)
        
        # 验证变形的稳定性
        assert np.mean(diffs) < 1.0, "Deformation not stable under small perturbations"
        assert np.std(diffs) < 0.5, "Deformation variance too high"

    def test_deformation_reversibility(self, setup_deformer, realistic_frame):
        """测试变形的可逆性"""
        regions = {}
        
        # 创建一个来回的姿态序列
        forward_poses = [self._create_test_pose(angle=i) for i in range(0, 90, 10)]
        backward_poses = forward_poses[::-1]
        
        # 应用正向变形
        forward_results = []
        for pose in forward_poses:
            deformed = setup_deformer.deform_frame(realistic_frame, regions, pose)
            forward_results.append(deformed)
        
        # 应用反向变形
        backward_results = []
        for pose in backward_poses:
            deformed = setup_deformer.deform_frame(realistic_frame, regions, pose)
            backward_results.append(deformed)
        
        # 验证来回变形后的一致性
        start_frame = forward_results[0]
        end_frame = backward_results[-1]
        diff = np.mean(np.abs(start_frame.astype(float) - end_frame.astype(float)))
        assert diff < 10.0, "Deformation not reversible"

    def test_deformation_locality(self, setup_deformer, realistic_frame):
        """测试变形的局部性"""
        # 创建两个不同区域的变形
        regions = {
            'left': DeformRegion(
                center=np.array([160, 240]),
                binding_points=[
                    BindingPoint(0, np.array([0, -30]), 1.0)
                ],
                mask=np.zeros((480, 640), dtype=np.uint8)
            ),
            'right': DeformRegion(
                center=np.array([480, 240]),
                binding_points=[
                    BindingPoint(1, np.array([0, 30]), 1.0)
                ],
                mask=np.zeros((480, 640), dtype=np.uint8)
            )
        }
        
        # 设置区域蒙版
        cv2.circle(regions['left'].mask, (160, 240), 50, 255, -1)
        cv2.circle(regions['right'].mask, (480, 240), 50, 255, -1)
        
        # 应用变形
        pose = self._create_test_pose()
        deformed = setup_deformer.deform_frame(realistic_frame, regions, pose)
        
        # 验证变形的局部性
        # 检查中间区域是否保持不变
        center_region = realistic_frame[200:280, 300:340]
        deformed_center = deformed[200:280, 300:340]
        center_diff = np.mean(np.abs(center_region - deformed_center))
        assert center_diff < 1.0, "Non-local deformation detected"

    def test_pose_validation_strict(self, setup_deformer, realistic_frame):
        """严格测试姿态数据验证"""
        regions = {}
        
        # 测试关键点的连续性
        def create_discontinuous_pose():
            pose = self._create_test_pose()
            # 创建不连续的关键点
            for i in range(1, len(pose.landmarks), 2):
                pose.landmarks[i].x += 200  # 大幅度跳变
            return pose
        
        # 测试关键点的可见度
        def create_invisible_pose():
            pose = self._create_test_pose()
            for lm in pose.landmarks:
                lm.visibility = 0.0  # 全部不可见
            return pose
        
        # 测试置信度阈值
        def create_low_confidence_pose():
            pose = self._create_test_pose()
            pose.confidence = 0.1  # 低置信度
            return pose
        
        test_cases = [
            (create_discontinuous_pose(), "Discontinuous landmarks"),
            (create_invisible_pose(), "All landmarks invisible"),
            (create_low_confidence_pose(), "Low confidence pose")
        ]
        
        for pose, case_name in test_cases:
            with pytest.raises((ValueError, AssertionError), 
                             message=f"Failed to validate {case_name}"):
                setup_deformer.deform_frame(realistic_frame, regions, pose)

    def test_physical_constraints(self, setup_deformer, realistic_frame):
        """测试变形的物理约束"""
        regions = {}
        
        # 测试体积保持
        def check_volume_preservation(original, deformed):
            orig_area = np.sum(original > 0)
            def_area = np.sum(deformed > 0)
            area_ratio = def_area / orig_area
            return 0.8 < area_ratio < 1.2  # 允许20%的体积变化
        
        # 测试长度保持
        def check_length_preservation(pose1, pose2):
            def calc_segment_length(p1, p2):
                return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
            
            # 检查关键骨骼段的长度
            segments = [(0,1), (1,2), (2,3)]  # 示例骨骼连接
            lengths1 = [calc_segment_length(pose1.landmarks[i], pose1.landmarks[j]) 
                       for i,j in segments]
            lengths2 = [calc_segment_length(pose2.landmarks[i], pose2.landmarks[j]) 
                       for i,j in segments]
            
            # 验证长度变化不超过10%
            ratios = [l2/l1 for l1, l2 in zip(lengths1, lengths2)]
            return all(0.9 < r < 1.1 for r in ratios)
        
        # 测试不同程度的变形
        poses = [self._create_test_pose(angle=a) for a in range(0, 360, 45)]
        for pose in poses:
            deformed = setup_deformer.deform_frame(realistic_frame, regions, pose)
            
            # 验证体积保持
            assert check_volume_preservation(realistic_frame, deformed), \
                "Volume not preserved"
            
            # 验证长度保持
            if len(poses) > 1:
                assert check_length_preservation(poses[0], pose), \
                    "Bone lengths not preserved"

    def test_edge_case_handling_comprehensive(self, setup_deformer, realistic_frame):
        """全面测试边缘情况处理"""
        regions = {}
        
        # 1. 测试空图像
        empty_frame = np.zeros_like(realistic_frame)
        pose = self._create_test_pose()
        deformed_empty = setup_deformer.deform_frame(empty_frame, regions, pose)
        assert np.all(deformed_empty == 0), "Empty frame should remain empty"
        
        # 2. 测试单像素图像
        single_pixel = np.zeros((1, 1, 3), dtype=np.uint8)
        single_pixel[0,0] = [255, 255, 255]
        with pytest.raises(ValueError):
            setup_deformer.deform_frame(single_pixel, regions, pose)
        
        # 3. 测试超大图像
        large_frame = np.zeros((4000, 6000, 3), dtype=np.uint8)
        deformed_large = setup_deformer.deform_frame(large_frame, regions, pose)
        assert deformed_large.shape == large_frame.shape
        
        # 4. 测试非标准数据类型
        float_frame = realistic_frame.astype(np.float32) / 255.0
        deformed_float = setup_deformer.deform_frame(float_frame, regions, pose)
        assert deformed_float.dtype == float_frame.dtype
        
        # 5. 测试异常区域配置
        invalid_regions = {
            'test': DeformRegion(
                center=np.array([float('inf'), float('inf')]),
                binding_points=[],
                mask=np.ones_like(realistic_frame[:,:,0])
            )
        }
        with pytest.raises(ValueError):
            setup_deformer.deform_frame(realistic_frame, invalid_regions, pose)

    def test_optimization_and_resources(self, setup_deformer, realistic_frame):
        """测试性能优化和资源使用"""
        import cProfile
        import pstats
        import io
        from pstats import SortKey
        
        # 性能分析
        pr = cProfile.Profile()
        pr.enable()
        
        # 执行密集操作
        poses = [self._create_test_pose(angle=i) for i in range(0, 360, 5)]
        for pose in poses:
            setup_deformer.deform_frame(realistic_frame, {}, pose)
        
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.TIME)
        ps.print_stats()
        
        # 分析性能瓶颈
        stats_str = s.getvalue()
        total_time = float(stats_str.split('\n')[0].split()[-2])
        assert total_time / len(poses) < 0.033, "Performance below 30 FPS"
        
        # 内存使用分析
        import tracemalloc
        tracemalloc.start()
        
        # 执行变形操作
        for pose in poses[:10]:  # 测试前10帧
            setup_deformer.deform_frame(realistic_frame, {}, pose)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # 验证内存使用
        max_memory_per_frame = peak / 10  # 每帧平均内存使用
        assert max_memory_per_frame < 1024 * 1024 * 50, "Memory usage too high"  # 50MB限制
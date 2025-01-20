import pytest
import numpy as np
import cv2
import time
import psutil
from pose.pose_binding import PoseBinding, BindingConfig
from pose.pose_data import PoseData

class TestPoseBinding:
    @pytest.fixture
    def setup_binding(self):
        """初始化测试环境"""
        config = BindingConfig(
            smoothing_factor=0.5,
            min_confidence=0.3,
            joint_limits={
                'shoulder': (-90, 90),
                'elbow': (0, 145),
                'knee': (0, 160)
            }
        )
        return PoseBinding(config)

    def test_initial_frame_binding(self, setup_binding):
        """测试初始帧绑定"""
        # 创建测试帧和姿态数据
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, (320, 240), 50, (255, 255, 255), -1)  # 添加一个标记
        
        pose_data = self._create_test_pose()
        
        # 测试绑定创建
        binding = setup_binding.create_binding(frame, pose_data)
        
        assert binding is not None
        assert binding.reference_frame.shape == frame.shape
        assert len(binding.landmarks) == len(pose_data.landmarks)
        assert binding.weights.shape[0] > 0
        assert binding.valid == True

    def test_mesh_deformation(self, setup_binding):
        """测试网格变形"""
        # 创建初始绑定
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        initial_pose = self._create_test_pose(angle=0)
        binding = setup_binding.create_binding(frame, initial_pose)
        
        # 测试不同角度的变形
        test_angles = [30, 60, 90]
        for angle in test_angles:
            new_pose = self._create_test_pose(angle=angle)
            deformed = setup_binding.deform_frame(binding, new_pose)
            
            assert deformed is not None
            assert deformed.shape == frame.shape
            # 验证变形后的图像与原图不同
            assert not np.array_equal(deformed, frame)

    def test_weight_computation(self, setup_binding):
        """测试权重计算"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pose_data = self._create_test_pose()
        binding = setup_binding.create_binding(frame, pose_data)
        
        # 验证权重
        weights = binding.weights
        assert weights.min() >= 0  # 权重非负
        assert weights.max() <= 1  # 权重不超过1
        assert np.allclose(weights.sum(axis=1), 1)  # 每个点的权重和为1

    def test_large_deformation(self, setup_binding):
        """测试大幅度变形"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        initial_pose = self._create_test_pose(angle=0)
        binding = setup_binding.create_binding(frame, initial_pose)
        
        # 测试大角度变形
        large_angle_pose = self._create_test_pose(angle=180)
        deformed = setup_binding.deform_frame(binding, large_angle_pose)
        
        assert deformed is not None
        # 验证变形是否保持图像完整性
        assert not np.any(np.isnan(deformed))
        assert not np.any(np.isinf(deformed))

    def test_confidence_threshold(self, setup_binding):
        """测试置信度阈值"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 测试低置信度
        low_conf_pose = self._create_test_pose(visibility=0.2)
        low_conf_binding = setup_binding.create_binding(frame, low_conf_pose)
        assert not low_conf_binding.valid
        
        # 测试边界置信度
        border_conf_pose = self._create_test_pose(visibility=0.3)
        border_conf_binding = setup_binding.create_binding(frame, border_conf_pose)
        assert border_conf_binding.valid
        
        # 测试高置信度
        high_conf_pose = self._create_test_pose(visibility=0.8)
        high_conf_binding = setup_binding.create_binding(frame, high_conf_pose)
        assert high_conf_binding.valid

    def test_joint_angle_limits(self, setup_binding):
        """测试关节角度限制"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        binding = setup_binding.create_binding(frame, self._create_test_pose())
        
        # 测试各个关节的限制
        test_cases = [
            ('shoulder', -100, -90),  # 超出下限
            ('shoulder', 100, 90),    # 超出上限
            ('elbow', -30, 0),        # 超出下限
            ('elbow', 160, 145),      # 超出上限
            ('knee', -20, 0),         # 超出下限
            ('knee', 180, 160)        # 超出上限
        ]
        
        for joint, input_angle, expected_angle in test_cases:
            result = setup_binding._apply_joint_limits({joint: input_angle})
            assert np.isclose(result[joint], expected_angle)

    def test_smoothing_behavior(self, setup_binding):
        """测试平滑行为"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        binding = setup_binding.create_binding(frame, self._create_test_pose())
        
        # 测试连续变形的平滑效果
        prev_deformed = None
        angles = range(0, 90, 10)
        
        for angle in angles:
            new_pose = self._create_test_pose(angle=float(angle))
            deformed = setup_binding.deform_frame(binding, new_pose)
            
            if prev_deformed is not None:
                # 计算相邻帧的差异
                diff = np.mean(np.abs(deformed - prev_deformed))
                # 验证变化是渐进的
                assert diff < 50  # 根据实际情况调整阈值
            
            prev_deformed = deformed.copy()

    def test_error_handling(self, setup_binding):
        """测试错误处理"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 测试无效的姿态数据
        with pytest.raises(ValueError):
            setup_binding.create_binding(frame, None)
        
        # 测试尺寸不匹配
        invalid_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        binding = setup_binding.create_binding(frame, self._create_test_pose())
        with pytest.raises(ValueError):
            setup_binding.deform_frame(binding, invalid_frame)
        
        # 测试关键点数量不匹配
        invalid_pose = PoseData(landmarks=[{
            'x': 0.5, 'y': 0.5, 'z': 0.0, 'visibility': 1.0
        }])  # 只有一个关键点
        with pytest.raises(ValueError):
            setup_binding.create_binding(frame, invalid_pose)

    def test_landmark_topology(self, setup_binding):
        """测试关键点拓扑结构"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pose = self._create_test_pose()
        binding = setup_binding.create_binding(frame, pose)
        
        # 验证骨骼拓扑
        topology = binding.get_topology()
        # 验证关键连接
        assert ('left_shoulder', 'left_elbow') in topology
        assert ('right_shoulder', 'right_elbow') in topology
        assert ('left_hip', 'left_knee') in topology
        assert ('right_hip', 'right_knee') in topology
        assert ('neck', 'spine') in topology

    def test_mesh_generation(self, setup_binding):
        """测试网格生成"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pose = self._create_test_pose()
        binding = setup_binding.create_binding(frame, pose)
        
        mesh = binding.get_mesh()
        # 验证网格点数量和分布
        assert mesh.shape[0] > 100  # 至少100个网格点
        assert mesh.shape[1] == 2   # x,y坐标
        
        # 验证网格覆盖范围
        assert np.all(mesh[:, 0] >= 0) and np.all(mesh[:, 0] < frame.shape[1])
        assert np.all(mesh[:, 1] >= 0) and np.all(mesh[:, 1] < frame.shape[0])

    def test_influence_weights(self, setup_binding):
        """测试影响权重"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pose = self._create_test_pose()
        binding = setup_binding.create_binding(frame, pose)
        
        # 获取特定点的权重
        test_points = [(320, 240), (100, 100), (500, 400)]
        for x, y in test_points:
            weights = binding.get_point_weights(x, y)
            # 验证权重和为1
            assert np.isclose(np.sum(weights), 1.0)
            # 验证权重非负
            assert np.all(weights >= 0)
            # 验证最近关键点权重最大
            nearest_idx = binding.find_nearest_landmark(x, y)
            assert weights[nearest_idx] == max(weights)

    def test_deformation_consistency(self, setup_binding):
        """测试变形一致性"""
        # 创建带有明显特征的测试帧
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        features = [
            (320, 240, 50),  # 中心圆
            (160, 120, 30),  # 左上圆
            (480, 360, 30)   # 右下圆
        ]
        for x, y, r in features:
            cv2.circle(frame, (x, y), r, (255, 255, 255), -1)
        
        initial_pose = self._create_test_pose(angle=0)
        binding = setup_binding.create_binding(frame, initial_pose)
        
        # 测试不同角度的变形
        test_angles = [30, 60, 90]
        for angle in test_angles:
            new_pose = self._create_test_pose(angle=angle)
            deformed = setup_binding.deform_frame(binding, new_pose)
            
            # 验证特征保持
            for x, y, r in features:
                # 计算变形后的特征位置
                transformed_pos = binding.transform_point(x, y, new_pose)
                # 在变形后的图像中查找特征
                roi = deformed[
                    max(0, int(transformed_pos[1]-r)):min(480, int(transformed_pos[1]+r)),
                    max(0, int(transformed_pos[0]-r)):min(640, int(transformed_pos[0]+r))
                ]
                # 验证特征存在（有白色像素）
                assert np.any(roi > 0)

    def test_temporal_coherence(self, setup_binding):
        """测试时间连贯性"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, (320, 240), 50, (255, 255, 255), -1)
        
        initial_pose = self._create_test_pose(angle=0)
        binding = setup_binding.create_binding(frame, initial_pose)
        
        prev_deformed = None
        # 测试连续小角度变化
        for angle in range(0, 30, 2):
            new_pose = self._create_test_pose(angle=float(angle))
            deformed = setup_binding.deform_frame(binding, new_pose)
            
            if prev_deformed is not None:
                # 计算相邻帧的差异
                diff = np.mean(np.abs(deformed - prev_deformed))
                # 验证变化平滑
                assert diff < 10  # 根据实际情况调整阈值
            
            prev_deformed = deformed.copy()

    def test_performance_metrics(self, setup_binding):
        """测试性能指标"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pose = self._create_test_pose()
        
        # 测试处理时间
        start_time = time.time()
        binding = setup_binding.create_binding(frame, pose)
        creation_time = time.time() - start_time
        
        start_time = time.time()
        deformed = setup_binding.deform_frame(binding, self._create_test_pose(angle=45))
        deform_time = time.time() - start_time
        
        # 验证性能在合理范围内
        assert creation_time < 0.1  # 创建绑定应在100ms内完成
        assert deform_time < 0.05   # 变形应在50ms内完成

    def test_memory_usage(self, setup_binding):
        """测试内存使用"""
        initial_memory = psutil.Process().memory_info().rss
        
        # 创建多个绑定和变形
        frames = []
        bindings = []
        for _ in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            binding = setup_binding.create_binding(frame, self._create_test_pose())
            frames.append(frame)
            bindings.append(binding)
        
        current_memory = psutil.Process().memory_info().rss
        memory_growth = (current_memory - initial_memory) / (1024 * 1024)  # MB
        
        # 验证内存增长在合理范围内
        assert memory_growth < 100  # 内存增长不应超过100MB

    def test_error_recovery(self, setup_binding):
        """测试错误恢复"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pose = self._create_test_pose()
        binding = setup_binding.create_binding(frame, pose)
        
        # 测试异常姿态的处理
        invalid_pose = self._create_test_pose()
        invalid_pose.landmarks[0]['x'] = float('inf')  # 制造无效数据
        
        # 应该能够优雅处理无效数据
        result = setup_binding.deform_frame(binding, invalid_pose)
        assert result is not None  # 应返回上一个有效结果

    def test_edge_cases(self, setup_binding):
        """测试边界条件"""
        # 测试空帧
        empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            setup_binding.create_binding(empty_frame, self._create_test_pose())
        
        # 测试异常姿态数据
        invalid_pose = self._create_test_pose()
        invalid_pose.landmarks[0]['x'] = float('inf')
        with pytest.raises(ValueError):
            setup_binding.create_binding(self.test_frame, invalid_pose)

    def test_concurrent_binding(self, setup_binding):
        """测试并发绑定"""
        import threading
        
        def bind_frame():
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            pose = self._create_test_pose()
            setup_binding.create_binding(frame, pose)
        
        threads = []
        for _ in range(5):
            t = threading.Thread(target=bind_frame)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()

    @staticmethod
    def _create_test_pose(angle: float = 0, visibility: float = 1.0) -> PoseData:
        """创建测试姿态数据
        
        Args:
            angle: 关节角度
            visibility: 关键点可见度
        """
        # 创建基本姿态
        landmarks = []
        for i in range(33):  # 使用标准的33个关键点
            x = 0.5 + 0.1 * np.cos(np.radians(angle))
            y = 0.5 + 0.1 * np.sin(np.radians(angle))
            landmarks.append({
                'x': x,
                'y': y,
                'z': 0.0,
                'visibility': visibility
            })
        
        return PoseData(landmarks=landmarks) 
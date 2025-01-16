import pytest
import numpy as np
from pose.pose_binding import PoseBinding, SkeletonBinding, Bone

class TestPoseBinding:
    @pytest.fixture
    def setup_binding(self):
        """初始化测试环境"""
        config = {
            'topology': [
                (0, 1), (1, 2), (2, 3),  # 脊柱
                (2, 4), (4, 5), (5, 6),  # 右臂
                (2, 7), (7, 8), (8, 9),  # 左臂
                (0, 10), (10, 11), (11, 12),  # 右腿
                (0, 13), (13, 14), (14, 15),  # 左腿
            ]
        }
        return PoseBinding(config)
        
    def test_create_binding(self, setup_binding):
        """测试骨骼绑定创建"""
        binding = setup_binding
        
        # 创建测试数据
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarks = [
            {'x': 0.5, 'y': 0.5, 'z': 0.0} 
            for _ in range(33)  # MediaPipe姿态点
        ]
        
        # 测试绑定创建
        result = binding.create_binding(frame, landmarks)
        
        assert isinstance(result, SkeletonBinding)
        assert len(result.bones) > 0
        assert result.weights.shape[0] == frame.shape[0] * frame.shape[1]
        
    def test_weight_computation(self, setup_binding):
        """测试权重计算"""
        binding = setup_binding
        
        # 创建测试点和骨骼
        points = np.random.rand(1000, 2)  # 1000个测试点
        bones = [
            Bone(0, 1, [2]),
            Bone(1, 2, [3]),
            Bone(2, 3, [])
        ]
        
        # 计算权重
        weights = binding.compute_weights(points, bones)
        
        assert weights.shape == (1000, len(bones))
        assert np.allclose(weights.sum(axis=1), 1.0)  # 权重和为1
        assert np.all(weights >= 0)  # 非负权重 
import pytest
import numpy as np
import time
from pose.pose_deformer import PoseDeformer
from pose.pose_binding import SkeletonBinding, Bone

class TestPoseDeformer:
    @pytest.fixture
    def setup_deformer(self):
        """初始化测试环境"""
        # 创建测试绑定
        binding = SkeletonBinding(
            landmarks=[{'x': 0.5, 'y': 0.5, 'z': 0.0} for _ in range(33)],
            bones=[Bone(0, 1, [2]), Bone(1, 2, []), Bone(2, 3, [])],
            weights=np.random.rand(1000, 3)  # 1000个点，3个骨骼
        )
        return PoseDeformer(binding)
        
    def test_transform_frame(self, setup_deformer):
        """测试帧变换"""
        deformer = setup_deformer
        
        # 创建测试数据
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        current_pose = [
            {'x': 0.6, 'y': 0.6, 'z': 0.0}  # 稍微移动的姿态
            for _ in range(33)
        ]
        
        # 测试变换性能
        start_time = time.time()
        result = deformer.transform_frame(frame, current_pose)
        transform_time = time.time() - start_time
        
        assert transform_time < 0.01  # 10ms内完成
        assert result.shape == frame.shape
        assert not np.array_equal(result, frame)  # 确保发生了变换
        
    def test_bone_transforms(self, setup_deformer):
        """测试骨骼变换"""
        deformer = setup_deformer
        
        # 创建测试姿态
        current_pose = [
            {'x': 0.5 + i*0.01, 'y': 0.5 + i*0.01, 'z': 0.0}
            for i in range(33)
        ]
        
        # 计算变换矩阵
        transforms = deformer.compute_bone_transforms(current_pose)
        
        assert len(transforms) == len(deformer.binding.bones)
        for transform in transforms:
            assert transform.shape == (4, 4)  # 齐次变换矩阵
            
    def test_deformation_stability(self, setup_deformer):
        """测试变形稳定性"""
        deformer = setup_deformer
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 创建轻微抖动的姿态序列
        results = []
        for i in range(10):
            noise = np.random.normal(0, 0.001, (33, 3))  # 小幅抖动
            pose = [
                {
                    'x': 0.5 + noise[j,0],
                    'y': 0.5 + noise[j,1],
                    'z': noise[j,2]
                }
                for j in range(33)
            ]
            result = deformer.transform_frame(frame, pose)
            results.append(result)
            
        # 检查相邻帧的差异
        diffs = []
        for i in range(1, len(results)):
            diff = np.abs(results[i] - results[i-1]).mean()
            diffs.append(diff)
            
        assert max(diffs) < 1.0  # 相邻帧差异不大 
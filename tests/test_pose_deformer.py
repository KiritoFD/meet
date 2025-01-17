import pytest
import numpy as np
from connect.pose_deformer import PoseDeformer
from connect.pose_protocol import PoseData

class TestPoseDeformer:
    @pytest.fixture
    def setup_deformer(self):
        """初始化测试环境"""
        return PoseDeformer()

    def test_pose_interpolation(self, setup_deformer):
        """测试姿态插值"""
        # 创建两个测试姿态
        pose1 = self._create_test_pose(0.0, 0.0)
        pose2 = self._create_test_pose(1.0, 1.0)
        
        # 测试0.5插值
        interpolated = setup_deformer.interpolate(pose1, pose2, 0.5)
        assert np.allclose(interpolated.landmarks[0]['x'], 0.5)
        assert np.allclose(interpolated.landmarks[0]['y'], 0.5)

    def test_pose_smoothing(self, setup_deformer):
        """测试姿态平滑"""
        # 创建带噪声的姿态序列
        noisy_poses = [
            self._create_test_pose(i + np.random.normal(0, 0.1))
            for i in range(10)
        ]
        
        # 应用平滑
        smoothed = setup_deformer.smooth_sequence(noisy_poses)
        
        # 验证平滑效果
        for i in range(1, len(smoothed)-1):
            curr_x = smoothed[i].landmarks[0]['x']
            prev_x = smoothed[i-1].landmarks[0]['x']
            next_x = smoothed[i+1].landmarks[0]['x']
            # 验证平滑度
            assert abs(curr_x - (prev_x + next_x)/2) < 0.1

    def test_pose_extrapolation(self, setup_deformer):
        """测试姿态预测"""
        # 创建历史姿态序列
        history = [
            self._create_test_pose(i * 0.1)
            for i in range(5)
        ]
        
        # 预测下一帧
        predicted = setup_deformer.predict_next(history)
        
        # 验证预测趋势
        assert predicted.landmarks[0]['x'] > history[-1].landmarks[0]['x']

    def test_error_handling(self, setup_deformer):
        """测试错误处理"""
        # 测试无效输入
        with pytest.raises(ValueError):
            setup_deformer.interpolate(None, None, 0.5)
            
        # 测试序列太短
        with pytest.raises(ValueError):
            setup_deformer.smooth_sequence([self._create_test_pose(0)])

    @staticmethod
    def _create_test_pose(x: float, y: float = None):
        """创建测试姿态数据"""
        if y is None:
            y = x
        return PoseData(
            landmarks=[{
                'x': x,
                'y': y,
                'z': 0.0,
                'visibility': 1.0
            }] * 33  # MediaPipe姿态点数量
        ) 
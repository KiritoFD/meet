import pytest
import numpy as np
import time
from unittest.mock import Mock
from pose.pose_data import PoseData
from connect.pose_sender import PoseSender

class TestPoseSender:
    @pytest.fixture
    def setup_sender(self):
        """初始化测试环境"""
        mock_socket = Mock()
        return PoseSender(mock_socket)

    def test_send_pose_data(self, setup_sender):
        """测试发送姿态数据"""
        pose_data = self._generate_test_pose()
        success = setup_sender.send_pose_data(
            room="test_room",
            pose_results=pose_data,
            timestamp=time.time()
        )
        assert success

    def test_compression_strategy(self, setup_sender):
        """测试压缩策略"""
        pose_data = self._generate_test_pose()
        
        setup_sender.enable_compression(True)
        compressed = setup_sender._compress_data(pose_data)
        
        setup_sender.enable_compression(False)
        uncompressed = setup_sender._get_data_size(pose_data)
        
        assert len(compressed) < uncompressed

    @staticmethod
    def _generate_test_pose():
        """生成测试姿态数据"""
        return PoseData(landmarks=[{
            'x': np.random.random(),
            'y': np.random.random(),
            'z': np.random.random(),
            'visibility': np.random.random()
        } for _ in range(33)]) 
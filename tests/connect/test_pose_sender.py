import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 调试信息
print("Python path:", sys.path)
print("Current file:", __file__)
print("Project root:", Path(__file__).parent.parent.parent)

import pytest
from unittest.mock import Mock, patch
import numpy as np
import time
from connect.pose_sender import PoseSender
from connect.socket_manager import SocketManager

class TestPoseSender:
    @pytest.fixture
    def mock_socket_manager(self):
        """创建模拟的SocketManager"""
        mock_manager = Mock(spec=SocketManager)
        mock_manager.connected = True
        return mock_manager

    @pytest.fixture
    def setup_sender(self, mock_socket_manager):
        """初始化测试环境"""
        sender = PoseSender(mock_socket_manager)
        return sender

    def test_send_pose_data(self, setup_sender, mock_socket_manager):
        """测试发送姿态数据"""
        # 创建测试数据
        pose_data = self._generate_test_pose()
        timestamp = time.time()

        # 测试发送
        success = setup_sender.send_pose_data(
            room="test_room",
            pose_results=pose_data,
            timestamp=timestamp
        )

        assert success
        mock_socket_manager.emit.assert_called_once()

    def test_send_performance(self, setup_sender):
        """测试发送性能"""
        frame_count = 100
        frame_times = []

        for _ in range(frame_count):
            start_time = time.time()
            setup_sender.send_pose_data(
                room="test_room",
                pose_results=self._generate_test_pose(),
                timestamp=time.time()
            )
            frame_times.append(time.time() - start_time)

        avg_time = sum(frame_times) / len(frame_times)
        assert avg_time < 0.005  # 每帧处理时间应小于5ms

    def test_error_handling(self, setup_sender, mock_socket_manager):
        """测试错误处理"""
        # 模拟发送失败
        mock_socket_manager.emit.side_effect = Exception("Send failed")

        success = setup_sender.send_pose_data(
            room="test_room",
            pose_results=self._generate_test_pose(),
            timestamp=time.time()
        )

        assert not success

    def test_queue_management(self, setup_sender):
        """测试队列管理"""
        # 设置较小的队列大小
        setup_sender.queue_size = 5
        
        # 快速发送多个帧
        for i in range(10):
            success = setup_sender.send_pose_data(
                room="test_room",
                pose_results=self._generate_test_pose(),
                timestamp=time.time()
            )
            if i >= 5:
                assert not success  # 队列已满

    def test_send_strategy(self, setup_sender):
        """测试发送策略"""
        # 测试帧率控制
        setup_sender.set_target_fps(30)
        frame_times = []
        
        for _ in range(100):
            start = time.time()
            setup_sender.send_pose_data(
                room="test_room",
                pose_results=self._generate_test_pose()
            )
            frame_times.append(time.time() - start)
        
        # 验证帧率
        avg_interval = sum(frame_times) / len(frame_times)
        assert abs(avg_interval - 1/30) < 0.005  # 允许5ms误差

    def test_compression_strategy(self, setup_sender):
        """测试压缩策略"""
        # 生成重复性高的数据
        repetitive_pose = self._generate_test_pose()
        
        # 测试自动压缩
        setup_sender.enable_compression(True)
        compressed_size = setup_sender._get_data_size(repetitive_pose)
        
        setup_sender.enable_compression(False)
        uncompressed_size = setup_sender._get_data_size(repetitive_pose)
        
        assert compressed_size < uncompressed_size

    @staticmethod
    def _generate_test_pose():
        """生成测试姿态数据"""
        return {
            'landmarks': [
                {
                    'x': np.random.random(),
                    'y': np.random.random(),
                    'z': np.random.random(),
                    'visibility': np.random.random()
                }
                for _ in range(33)  # MediaPipe姿态点数量
            ]
        } 
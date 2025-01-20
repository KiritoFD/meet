import pytest
import numpy as np
import random
from unittest.mock import Mock
from connect.socket_manager import SocketManager
from connect.unified_sender import UnifiedSender
from connect.performance_monitor import PerformanceMonitor

@pytest.fixture
def mock_socket():
    """创建模拟的Socket连接"""
    socket = Mock(spec=SocketManager)
    socket.connected = True
    socket.emit = Mock(return_value=True)
    return socket

@pytest.fixture
def performance_monitor():
    """创建性能监控器"""
    return PerformanceMonitor()

@pytest.fixture
def unified_sender(mock_socket, performance_monitor):
    """创建统一发送器"""
    return UnifiedSender(mock_socket, performance_monitor)

def generate_test_pose(landmark_count: int = 33) -> dict:
    """生成测试姿态数据"""
    return {
        'landmarks': [
            {
                'x': random.random(),
                'y': random.random(),
                'z': random.random(),
                'visibility': random.random()
            }
            for _ in range(landmark_count)
        ]
    }

def generate_test_audio(duration: float = 1.0) -> bytes:
    """生成测试音频数据"""
    sample_rate = 44100
    samples = np.random.random(int(duration * sample_rate))
    return samples.tobytes() 
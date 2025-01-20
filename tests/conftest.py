import os
import sys
from pathlib import Path
import pytest
import logging
import numpy as np
import cv2

# 配置项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@pytest.fixture(scope="session")
def test_frame():
    """创建测试用的图像帧"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # 添加一些特征以便测试
    cv2.circle(frame, (320, 240), 50, (255, 255, 255), -1)
    cv2.rectangle(frame, (100, 100), (200, 200), (128, 128, 128), -1)
    return frame

@pytest.fixture(scope="session")
def test_pose_data():
    """创建测试用的姿态数据"""
    return {
        'landmarks': [
            {'x': 0.5, 'y': 0.5, 'z': 0.0, 'visibility': 1.0}
            for _ in range(33)  # MediaPipe标准关键点数量
        ]
    }

@pytest.fixture(scope="function")
def mock_socket():
    """创建模拟的Socket对象"""
    from unittest.mock import Mock
    socket = Mock()
    socket.emit = Mock(return_value=True)
    socket.connected = True
    return socket

# 打印调试信息
print("Added to Python path:", project_root)
print("Current sys.path:", sys.path)

@pytest.fixture(autouse=True)
def setup_test_env():
    """设置测试环境"""
    # 这里可以添加其他测试环境设置
    pass 
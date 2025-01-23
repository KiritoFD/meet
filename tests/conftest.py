import os
import sys
from pathlib import Path
import pytest
import logging
import numpy as np
import cv2

# 获取项目根目录并添加到 Python 路径
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 打印调试信息
print(f"Current working directory: {os.getcwd()}")
print(f"Project root: {project_root}")
print(f"Python path: {sys.path}")

# 创建输出目录
output_dir = os.path.join(project_root, 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 配置日志输出到文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(output_dir, 'test.log')),
        logging.StreamHandler()
    ]
)

def pytest_configure(config):
    """配置pytest"""
    # 添加输出目录到pytest配置
    config.option.output_dir = output_dir
    
    # 初始化metadata字典
    if not hasattr(config, '_metadata'):
        config._metadata = {}
    
    # 添加项目信息到metadata
    config._metadata.update({
        'Project': 'Avatar System',
        'output_dir': str(output_dir)
    })

    # 配置HTML报告
    if hasattr(config, '_html'):
        config._html.style_css = '''
            body { font-family: Arial, sans-serif; }
            h2 { color: #2C3E50; }
            .passed { color: #27AE60; }
            .failed { color: #E74C3C; }
            .skipped { color: #F39C12; }
        '''

def pytest_sessionstart(session):
    """测试会话开始时的处理"""
    # 清理旧的测试报告
    for file in os.listdir(output_dir):
        if file.endswith('.xml') or file.endswith('.html'):
            os.remove(os.path.join(output_dir, file))

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

@pytest.fixture(autouse=True)
def setup_test_env():
    """设置测试环境"""
    # 这里可以添加其他测试环境设置
    pass 
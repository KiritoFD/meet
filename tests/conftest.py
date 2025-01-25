import os
import sys
from pathlib import Path

# 获取项目根目录并添加到 Python 路径
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
import logging
import numpy as np
import random
from unittest.mock import Mock, AsyncMock
from typing import Dict, Optional

# 延迟导入
def get_jwt():
    try:
        import jwt
        return jwt
    except ImportError:
        return None

def get_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        return None

@pytest.fixture
def jwt():
    """提供jwt模块"""
    jwt = get_jwt()
    if jwt is None:
        pytest.skip("PyJWT not available")
    return jwt

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

@pytest.fixture(scope="session")
def test_frame(cv2):
    """创建测试用的图像帧"""
    if cv2 is None:
        pytest.skip("OpenCV (cv2) not available")
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

@pytest.fixture
def cv2():
    """提供cv2模块，如果不可用则跳过测试"""
    cv2 = get_cv2()
    if cv2 is None:
        pytest.skip("OpenCV (cv2) not available")
    return cv2

@pytest.fixture
def config():
    return {
        'sender': {
            'queue_size': 1000,
            'batch_size': 10,
            'send_interval': 0.1
        },
        'processor': {
            'compression_level': 6,
            'max_pose_size': 1024 * 1024,
            'batch_size': 10
        },
        'jwt': {
            'secret_key': 'test_secret_key',
            'token_expiry': 3600
        }
    }

@pytest.fixture
def jwt_handler(config):
    """创建JWT处理器"""
    from lib.jwt_utils import JWTHandler
    return JWTHandler(config['jwt']['secret_key'])

@pytest.fixture
def auth_token(jwt_handler):
    """创建测试用的认证令牌"""
    return jwt_handler.generate_token('test_user')

@pytest.fixture
def mock_socket():
    socket = Mock()
    socket.connected = True
    socket.emit = Mock(return_value=True)
    return socket

@pytest.fixture(scope="session")
def event_loop_policy():
    """提供事件循环策略"""
    import asyncio
    return asyncio.WindowsSelectorEventLoopPolicy()

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

def pytest_sessionstart(session):
    """测试会话开始时的处理"""
    # 清理旧的测试报告
    for file in os.listdir(output_dir):
        if file.endswith('.xml') or file.endswith('.html'):
            os.remove(os.path.join(output_dir, file))

@pytest.fixture(autouse=True)
def setup_test_env():
    """设置测试环境"""
    # 这里可以添加其他测试环境设置
    pass
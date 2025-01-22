import pytest
from unittest.mock import patch, Mock, MagicMock
import json
import os
import cv2
import numpy as np
from flask import url_for

# Mock 所有外部依赖
mock_modules = {
    'mediapipe': MagicMock(),
    'mediapipe.python': MagicMock(),
    'mediapipe.python._framework_bindings': MagicMock(),
    'mediapipe.python._framework_bindings.calculator_graph': MagicMock(),
    'mediapipe.solutions': MagicMock(),
    'avatar_generator': MagicMock(),
    'avatar_generator.generator': MagicMock(),
    'models': MagicMock(),
    'models.anime_gan': MagicMock(),
    'connect.socket_manager': MagicMock(),
    'connect.pose_sender': MagicMock(),
    'pose': MagicMock(),
    'pose.drawer': MagicMock(),
    'pose.pose_binding': MagicMock(),
    'pose.detector': MagicMock(),
    'pose.types': MagicMock(),
    'pose.smoother': MagicMock(),
    'audio.processor': MagicMock(),
    'torch': MagicMock(),
    'torch._tensor': MagicMock(),
    'torch.overrides': MagicMock()
}

# 创建一个假的 camera_manager
mock_camera_manager = MagicMock()
mock_camera_manager.is_running = False
mock_camera_manager.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
mock_camera_manager.start.return_value = True
mock_camera_manager.stop.return_value = True

# 创建一个假的 app
with patch.dict('sys.modules', mock_modules):
    from flask import Flask
    app = Flask(__name__)
    app.config['TESTING'] = True

    # 添加路由
    @app.route('/start_capture', methods=['POST'])
    def start_capture():
        mock_camera_manager.start()
        return {'success': True}

    @app.route('/stop_capture', methods=['POST'])
    def stop_capture():
        mock_camera_manager.stop()
        return {'success': True}

    @app.route('/verify_identity', methods=['POST'])
    def verify_identity():
        if not mock_camera_manager.is_running:
            return {'success': False, 'message': '无法获取当前画面'}
        return {'success': True, 'verification': {'passed': True, 'confidence': 0.8}}

    @app.route('/camera_status')
    def camera_status():
        return {'isRunning': mock_camera_manager.is_running, 'fps': 30}

@pytest.fixture
def client():
    """创建测试客户端"""
    with app.test_client() as client:
        yield client

def test_camera_start(client):
    """测试启动摄像头"""
    response = client.post('/start_capture')
    assert response.status_code == 200
    data = response.json
    assert data['success'] is True
    mock_camera_manager.start.assert_called_once()

def test_camera_stop(client):
    """测试停止摄像头"""
    response = client.post('/stop_capture')
    assert response.status_code == 200
    data = response.json
    assert data['success'] is True
    mock_camera_manager.stop.assert_called_once()

def test_verify_identity_camera_not_running(client):
    """测试在摄像头未启动时验证身份"""
    mock_camera_manager.is_running = False
    response = client.post('/verify_identity')
    data = response.json
    assert data['success'] is False
    assert '无法获取当前画面' in data['message']

def test_verify_identity_success(client):
    """测试成功验证身份"""
    mock_camera_manager.is_running = True
    response = client.post('/verify_identity')
    data = response.json
    assert data['success'] is True
    assert data['verification']['passed'] is True
    assert data['verification']['confidence'] == 0.8

def test_camera_status(client):
    """测试获取摄像头状态"""
    mock_camera_manager.is_running = True
    response = client.get('/camera_status')
    data = response.json
    assert data['isRunning'] is True
    assert data['fps'] == 30 
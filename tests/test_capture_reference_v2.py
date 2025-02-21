import pytest
import numpy as np
import cv2
import time
from unittest.mock import Mock, patch
from flask import jsonify
from run import app, capture_reference

@pytest.fixture
def mock_frame():
    """创建模拟的图像帧"""
    return np.zeros((480, 640, 3), dtype=np.uint8)

@pytest.fixture
def mock_pose_data():
    """创建模拟的姿态数据"""
    class MockPoseData:
        def __init__(self):
            self.landmarks = []
            for _ in range(33):
                lm = Mock()
                lm.x = 0.5
                lm.y = 0.5
                lm.z = 0.0
                lm.visibility = 0.9
                self.landmarks.append(lm)
            self.face_landmarks = [Mock() for _ in range(468)]
            for lm in self.face_landmarks:
                lm.x = 0.5
                lm.y = 0.2
                lm.z = 0.0
                lm.visibility = 0.9
            self.timestamp = time.time()
            self.confidence = 0.9

        def __iter__(self):
            return iter(self.landmarks)

    return MockPoseData()

@pytest.fixture
def mock_camera_manager():
    """模拟摄像头管理器"""
    manager = Mock()
    manager.is_running = True
    manager.read_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    return manager

@pytest.fixture
def mock_pose():
    """模拟姿态检测器"""
    pose = Mock()
    return pose

@pytest.fixture
def mock_face_mesh():
    """模拟面部网格检测器"""
    face_mesh = Mock()
    face_mesh.process.return_value = Mock(multi_face_landmarks=[Mock()])
    return face_mesh

@pytest.fixture
def mock_pose_binding():
    """模拟姿态绑定器"""
    binder = Mock()
    binder.create_binding.return_value = [Mock(type='body'), Mock(type='face')]
    return binder

class TestCaptureReference:
    def test_successful_capture(self, mock_camera_manager, mock_pose, mock_face_mesh, 
                              mock_pose_binding, mock_frame, mock_pose_data):
        """测试成功捕获参考帧的情况"""
        mock_camera_manager.read_frame.return_value = mock_frame
        
        # 设置姿态检测结果
        pose_results = Mock()
        pose_results.pose_landmarks = Mock()
        pose_results.pose_landmarks.landmark = mock_pose_data.landmarks
        mock_pose.process.return_value = pose_results
        
        # 设置面部检测结果
        face_results = Mock()
        face_results.multi_face_landmarks = [Mock()]
        face_results.multi_face_landmarks[0].landmark = mock_pose_data.face_landmarks
        mock_face_mesh.process.return_value = face_results
        
        # 设置姿态绑定结果
        mock_pose_binding.create_binding.return_value = [Mock(type='body'), Mock(type='face')]
        
        with app.app_context():
            with patch('run.camera_manager', mock_camera_manager), \
                 patch('run.pose', mock_pose), \
                 patch('run.face_mesh', mock_face_mesh), \
                 patch('run.pose_binding', mock_pose_binding):
                
                response = capture_reference()
                if isinstance(response, tuple):
                    response, status_code = response
                response_data = response.get_json()
                
                assert status_code == 200
                assert response_data['success'] is True
                assert 'regions_info' in response_data['details']
                assert response_data['details']['regions_info']['body'] == 1
                assert response_data['details']['regions_info']['face'] == 1
    
    def test_camera_not_running(self, mock_camera_manager):
        """测试摄像头未运行的情况"""
        mock_camera_manager.is_running = False
        
        with app.app_context():
            with patch('run.camera_manager', mock_camera_manager):
                response = capture_reference()
                if isinstance(response, tuple):
                    response, status_code = response
                response_data = response.get_json()
                
                assert status_code == 400
                assert response_data['success'] is False
                assert '摄像头未运行' in response_data['message']
    
    def test_invalid_frame(self, mock_camera_manager, mock_pose):
        """测试无效帧的情况"""
        mock_camera_manager.read_frame.return_value = np.array([])
        
        with app.app_context():
            with patch('run.camera_manager', mock_camera_manager), \
                 patch('run.pose', mock_pose):
                response = capture_reference()
                if isinstance(response, tuple):
                    response, status_code = response
                response_data = response.get_json()
                
                assert status_code == 500
                assert response_data['success'] is False
                assert '无效的摄像头画面' in response_data['message']
    
    def test_no_pose_detected(self, mock_camera_manager, mock_pose):
        """测试未检测到姿态的情况"""
        mock_pose.process.return_value = Mock(pose_landmarks=None)
        
        with app.app_context():
            with patch('run.camera_manager', mock_camera_manager), \
                 patch('run.pose', mock_pose):
                response = capture_reference()
                if isinstance(response, tuple):
                    response, status_code = response
                response_data = response.get_json()
                
                assert status_code == 400
                assert response_data['success'] is False
                assert '未检测到人物姿态' in response_data['message']
    
    def test_no_face_detected(self, mock_camera_manager, mock_pose, mock_face_mesh):
        """测试未检测到面部的情况"""
        mock_pose.process.return_value.pose_landmarks.landmark = [Mock() for _ in range(33)]
        mock_face_mesh.process.return_value = Mock(multi_face_landmarks=[])
        
        with app.app_context():
            with patch('run.camera_manager', mock_camera_manager), \
                 patch('run.pose', mock_pose), \
                 patch('run.face_mesh', mock_face_mesh):
                response = capture_reference()
                if isinstance(response, tuple):
                    response, status_code = response
                response_data = response.get_json()
                
                assert status_code == 400
                assert response_data['success'] is False
                assert '未检测到面部' in response_data['message']
    
    def test_insufficient_landmarks(self, mock_camera_manager, mock_pose, mock_face_mesh):
        """测试关键点不足的情况"""
        # 设置姿态检测结果
        pose_results = Mock()
        pose_results.pose_landmarks = Mock()
        landmarks = [Mock() for _ in range(15)]  # 只有15个关键点
        for lm in landmarks:
            lm.x = 0.5
            lm.y = 0.5
            lm.z = 0.0
            lm.visibility = 0.9
        pose_results.pose_landmarks.landmark = landmarks
        mock_pose.process.return_value = pose_results
        
        # 设置面部检测结果
        face_results = Mock()
        face_results.multi_face_landmarks = [Mock()]
        face_results.multi_face_landmarks[0].landmark = [Mock() for _ in range(468)]
        mock_face_mesh.process.return_value = face_results
        
        with app.app_context():
            with patch('run.camera_manager', mock_camera_manager), \
                 patch('run.pose', mock_pose), \
                 patch('run.face_mesh', mock_face_mesh):
                response = capture_reference()
                if isinstance(response, tuple):
                    response, status_code = response
                response_data = response.get_json()
                
                assert status_code == 400
                assert response_data['success'] is False
                assert '检测到的关键点不完整' in response_data['message']
    
    def test_low_visibility_landmarks(self, mock_camera_manager, mock_pose, mock_face_mesh):
        """测试关键点可见度过低的情况"""
        landmarks = [Mock() for _ in range(33)]
        for lm in landmarks:
            lm.visibility = 0.1  # 设置低可见度
        mock_pose.process.return_value.pose_landmarks.landmark = landmarks
        
        with app.app_context():
            with patch('run.camera_manager', mock_camera_manager), \
                 patch('run.pose', mock_pose), \
                 patch('run.face_mesh', mock_face_mesh):
                response = capture_reference()
                if isinstance(response, tuple):
                    response, status_code = response
                response_data = response.get_json()
                
                assert status_code == 400
                assert response_data['success'] is False
                assert '姿态检测置信度过低' in response_data['message']
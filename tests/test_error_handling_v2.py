import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch
from run import app, capture_reference

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
    pose.process.return_value = Mock(pose_landmarks=Mock())
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
    binding = Mock()
    binding.create_binding.return_value = [Mock(type='body'), Mock(type='face')]
    return binding

class TestErrorHandling:
    def test_camera_exception(self, mock_camera_manager):
        """测试摄像头异常的情况"""
        mock_camera_manager.read_frame.side_effect = Exception("Camera error")
        
        with app.app_context():
            with patch('run.camera_manager', mock_camera_manager):
                response = capture_reference()
                if isinstance(response, tuple):
                    response, status_code = response
                response_data = response.get_json()
                
                assert status_code == 500
                assert response_data['success'] is False
                assert 'Camera error' in response_data['message']
    
    def test_pose_detection_exception(self, mock_camera_manager, mock_pose):
        """测试姿态检测异常的情况"""
        mock_pose.process.side_effect = Exception("Pose detection error")
        
        with app.app_context():
            with patch('run.camera_manager', mock_camera_manager), \
                 patch('run.pose', mock_pose):
                response = capture_reference()
                if isinstance(response, tuple):
                    response, status_code = response
                response_data = response.get_json()
                
                assert status_code == 500
                assert response_data['success'] is False
                assert 'Pose detection error' in response_data['message']
    
    def test_face_detection_exception(self, mock_camera_manager, mock_pose, mock_face_mesh):
        """测试面部检测异常的情况"""
        mock_face_mesh.process.side_effect = Exception("Face detection error")
        
        with app.app_context():
            with patch('run.camera_manager', mock_camera_manager), \
                 patch('run.pose', mock_pose), \
                 patch('run.face_mesh', mock_face_mesh):
                response = capture_reference()
                if isinstance(response, tuple):
                    response, status_code = response
                response_data = response.get_json()
                
                assert status_code == 500
                assert response_data['success'] is False
                assert 'Face detection error' in response_data['message']
    
    def test_binding_exception(self, mock_camera_manager, mock_pose, mock_face_mesh, mock_pose_binding):
        """测试姿态绑定异常的情况"""
        mock_pose_binding.create_binding.side_effect = Exception("Binding error")
        
        with app.app_context():
            with patch('run.camera_manager', mock_camera_manager), \
                 patch('run.pose', mock_pose), \
                 patch('run.face_mesh', mock_face_mesh), \
                 patch('run.pose_binding', mock_pose_binding):
                response = capture_reference()
                if isinstance(response, tuple):
                    response, status_code = response
                response_data = response.get_json()
                
                assert status_code == 500
                assert response_data['success'] is False
                assert 'Binding error' in response_data['message']
    
    def test_invalid_frame_shape(self, mock_camera_manager):
        """测试无效的帧形状"""
        mock_camera_manager.read_frame.return_value = np.zeros((480, 640), dtype=np.uint8)  # 缺少通道维度
        
        with app.app_context():
            with patch('run.camera_manager', mock_camera_manager):
                response = capture_reference()
                if isinstance(response, tuple):
                    response, status_code = response
                response_data = response.get_json()
                
                assert status_code == 500
                assert response_data['success'] is False
                assert '无效的摄像头画面' in response_data['message']
    
    def test_corrupted_frame(self, mock_camera_manager):
        """测试损坏的帧数据"""
        mock_camera_manager.read_frame.return_value = np.array([1, 2, 3])  # 不正确的数组形状
        
        with app.app_context():
            with patch('run.camera_manager', mock_camera_manager):
                response = capture_reference()
                if isinstance(response, tuple):
                    response, status_code = response
                response_data = response.get_json()
                
                assert status_code == 500
                assert response_data['success'] is False
                assert '无效的摄像头画面' in response_data['message']
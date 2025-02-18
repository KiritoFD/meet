import pytest
import cv2
import numpy as np
import os
from unittest.mock import patch, Mock
from face.face_verification import FaceVerifier, FaceVerificationResult

@pytest.fixture
def mock_face_recognition():
    """Mock face_recognition 库"""
    with patch('face.face_verification.face_recognition') as mock_fr:
        # 模拟人脸位置检测
        mock_fr.face_locations.return_value = [(100, 200, 300, 100)]  # top, right, bottom, left
        # 模拟人脸编码
        mock_fr.face_encodings.return_value = [np.random.rand(128)]
        # 模拟人脸距离计算
        mock_fr.face_distance.return_value = [0.3]  # 距离小于0.6表示匹配
        yield mock_fr

@pytest.fixture
def verifier(mock_face_recognition):
    """创建带有mock的FaceVerifier实例"""
    return FaceVerifier(similarity_threshold=0.6)

@pytest.fixture
def sample_image():
    """创建测试图片"""
    return np.zeros((480, 640, 3), dtype=np.uint8)

def test_face_verifier_init(verifier):
    """测试FaceVerifier初始化"""
    assert verifier.similarity_threshold == 0.6
    assert verifier.reference_encoding is None

def test_set_reference_success(verifier, sample_image, mock_face_recognition):
    """测试成功设置参考帧"""
    success = verifier.set_reference(sample_image)
    assert success
    assert verifier.reference_encoding is not None
    mock_face_recognition.face_locations.assert_called_once()
    mock_face_recognition.face_encodings.assert_called_once()

def test_set_reference_no_face(verifier, sample_image, mock_face_recognition):
    """测试设置参考帧时没有检测到人脸"""
    mock_face_recognition.face_locations.return_value = []
    success = verifier.set_reference(sample_image)
    assert not success
    assert verifier.reference_encoding is None

def test_verify_face_no_reference(verifier, sample_image):
    """测试在没有参考帧的情况下验证"""
    result = verifier.verify_face(sample_image)
    assert isinstance(result, FaceVerificationResult)
    assert not result.is_same_person
    assert result.error_message == "No reference face set"

def test_verify_face_match(verifier, sample_image, mock_face_recognition):
    """测试人脸匹配的情况"""
    # 设置参考帧
    verifier.set_reference(sample_image)
    # 设置较小的距离表示匹配
    mock_face_recognition.face_distance.return_value = [0.3]
    
    result = verifier.verify_face(sample_image)
    assert isinstance(result, FaceVerificationResult)
    assert result.is_same_person
    assert result.confidence > 0.6

def test_verify_face_no_match(verifier, sample_image, mock_face_recognition):
    """测试人脸不匹配的情况"""
    # 设置参考帧
    verifier.set_reference(sample_image)
    # 设置较大的距离表示不匹配
    mock_face_recognition.face_distance.return_value = [0.8]
    
    result = verifier.verify_face(sample_image)
    assert isinstance(result, FaceVerificationResult)
    assert not result.is_same_person
    assert result.confidence < 0.6

def test_verify_face_no_face_detected(verifier, sample_image, mock_face_recognition):
    """测试没有检测到人脸的情况"""
    verifier.set_reference(sample_image)
    mock_face_recognition.face_locations.return_value = []
    
    result = verifier.verify_face(sample_image)
    assert isinstance(result, FaceVerificationResult)
    assert not result.is_same_person
    assert "No face detected" in result.error_message

def test_get_face_location(verifier, sample_image, mock_face_recognition):
    """测试获取人脸位置"""
    expected_location = (100, 200, 300, 100)
    mock_face_recognition.face_locations.return_value = [expected_location]
    
    location = verifier.get_face_location(sample_image)
    assert location == expected_location
    mock_face_recognition.face_locations.assert_called_once()

def test_get_face_location_no_face(verifier, sample_image, mock_face_recognition):
    """测试获取人脸位置时没有检测到人脸"""
    mock_face_recognition.face_locations.return_value = []
    
    location = verifier.get_face_location(sample_image)
    assert location is None

def test_draw_face_box_with_face(verifier, sample_image):
    """测试绘制人脸框（有人脸）"""
    result = FaceVerificationResult(is_same_person=True, confidence=0.8)
    location = (100, 200, 300, 100)
    
    with patch.object(verifier, 'get_face_location', return_value=location):
        output = verifier.draw_face_box(sample_image, result)
        assert output.shape == sample_image.shape
        assert not np.array_equal(output, sample_image)

def test_draw_face_box_no_face(verifier, sample_image):
    """测试绘制人脸框（无人脸）"""
    result = FaceVerificationResult(is_same_person=False, confidence=0.0)
    
    with patch.object(verifier, 'get_face_location', return_value=None):
        output = verifier.draw_face_box(sample_image, result)
        assert output.shape == sample_image.shape
        assert np.array_equal(output, sample_image)

def test_clear_reference(verifier):
    """测试清除参考帧"""
    verifier.reference_encoding = np.zeros(128)
    assert verifier.reference_encoding is not None
    verifier.clear_reference()
    assert verifier.reference_encoding is None 
import pytest
import numpy as np
from pose.pose_data import PoseData, Landmark
from to_center import center_pose
from config.settings import POSE_CONFIG

def create_test_landmarks(num_points=33):
    """创建测试用的关键点列表"""
    return [Landmark(x=0.0, y=0.0, z=0.0, visibility=0.0) for _ in range(num_points)]

def test_center_pose_normal():
    """测试正常姿态数据的居中"""
    landmarks = create_test_landmarks()
    keypoints = POSE_CONFIG['detector']['keypoints']
    
    # 设置关键躯干点
    landmarks[keypoints['nose']['id']] = Landmark(x=0.3, y=0.3, z=0, visibility=1.0)
    landmarks[keypoints['neck']['id']] = Landmark(x=0.35, y=0.4, z=0, visibility=1.0)
    landmarks[keypoints['left_shoulder']['id']] = Landmark(x=0.3, y=0.4, z=0, visibility=1.0)
    landmarks[keypoints['right_shoulder']['id']] = Landmark(x=0.4, y=0.4, z=0, visibility=1.0)
    landmarks[keypoints['left_hip']['id']] = Landmark(x=0.3, y=0.6, z=0, visibility=1.0)
    landmarks[keypoints['right_hip']['id']] = Landmark(x=0.4, y=0.6, z=0, visibility=1.0)
    
    test_pose = PoseData(landmarks=landmarks, timestamp=0, confidence=1.0)
    centered_pose = center_pose(test_pose)
    
    # 验证躯干中心点是否在0.5,0.5位置
    trunk_ids = [
        keypoints['left_shoulder']['id'],
        keypoints['right_shoulder']['id'],
        keypoints['left_hip']['id'],
        keypoints['right_hip']['id']
    ]
    
    trunk_points = np.array([[centered_pose.landmarks[i].x, centered_pose.landmarks[i].y] 
                            for i in trunk_ids])
    center = trunk_points.mean(axis=0)
    np.testing.assert_almost_equal(center, [0.5, 0.5], decimal=6)

def test_center_pose_missing_points():
    """测试缺失关键点的情况"""
    landmarks = create_test_landmarks()
    keypoints = POSE_CONFIG['detector']['keypoints']
    
    # 只设置部分关键点
    landmarks[keypoints['nose']['id']] = Landmark(x=0.3, y=0.3, z=0, visibility=1.0)
    landmarks[keypoints['neck']['id']] = Landmark(x=0.35, y=0.4, z=0, visibility=1.0)
    # 其他点保持不可见
    
    test_pose = PoseData(landmarks=landmarks, timestamp=0, confidence=1.0)
    centered_pose = center_pose(test_pose)
    
    # 验证可见点是否被正确居中
    visible_points = np.array([
        [landmarks[keypoints['nose']['id']].x, landmarks[keypoints['nose']['id']].y],
        [landmarks[keypoints['neck']['id']].x, landmarks[keypoints['neck']['id']].y]
    ])
    
    center = visible_points.mean(axis=0)
    offset = np.array([0.5, 0.5]) - center
    
    np.testing.assert_almost_equal(
        [centered_pose.landmarks[keypoints['nose']['id']].x,
         centered_pose.landmarks[keypoints['nose']['id']].y],
        [landmarks[keypoints['nose']['id']].x + offset[0],
         landmarks[keypoints['nose']['id']].y + offset[1]],
        decimal=6
    )

def test_center_pose_empty():
    """测试空姿态数据"""
    empty_pose = PoseData(landmarks=[], timestamp=0, confidence=0.0)
    result = center_pose(empty_pose)
    assert result == empty_pose

def test_center_pose_low_visibility():
    """测试低可见度关键点"""
    test_landmarks = [
        Landmark(x=0.3, y=0.3, z=0, visibility=0.1),  # 低可见度
        Landmark(x=0.4, y=0.4, z=0, visibility=0.8),  # 高可见度
        Landmark(x=0.5, y=0.5, z=0, visibility=0.9),  # 高可见度
    ]
    test_pose = PoseData(landmarks=test_landmarks, timestamp=0, confidence=1.0)
    
    centered_pose = center_pose(test_pose)
    
    # 只考虑高可见度点的中心
    high_vis_coords = np.array([
        [lm.x, lm.y] for lm in test_landmarks[1:]  # 只取后两个点
    ])
    target_center = np.array([0.5, 0.5])
    center = high_vis_coords.mean(axis=0)
    offset = target_center - center
    
    # 验证所有点都被正确移动
    for orig, centered in zip(test_pose.landmarks, centered_pose.landmarks):
        np.testing.assert_almost_equal(
            [centered.x, centered.y],
            [orig.x + offset[0], orig.y + offset[1]],
            decimal=6
        )

def test_center_pose_edge_cases():
    """测试边界情况"""
    # 所有点都在同一位置
    test_landmarks = [
        Landmark(x=0.1, y=0.1, z=0, visibility=1.0),
        Landmark(x=0.1, y=0.1, z=0, visibility=1.0),
    ]
    test_pose = PoseData(landmarks=test_landmarks, timestamp=0, confidence=1.0)
    
    centered_pose = center_pose(test_pose)
    
    # 验证所有点都被移动到中心
    for lm in centered_pose.landmarks:
        np.testing.assert_almost_equal([lm.x, lm.y], [0.5, 0.5], decimal=6)

def test_center_pose_preserve_z():
    """测试Z坐标保持不变"""
    test_landmarks = [
        Landmark(x=0.3, y=0.3, z=1.5, visibility=1.0),
        Landmark(x=0.4, y=0.4, z=-0.5, visibility=1.0),
    ]
    test_pose = PoseData(landmarks=test_landmarks, timestamp=0, confidence=1.0)
    
    centered_pose = center_pose(test_pose)
    
    # 验证Z坐标保持不变
    for orig, centered in zip(test_pose.landmarks, centered_pose.landmarks):
        assert orig.z == centered.z

def test_center_pose_preserve_metadata():
    """测试元数据保持不变"""
    timestamp = 12345
    confidence = 0.95
    test_landmarks = [
        Landmark(x=0.3, y=0.3, z=0, visibility=1.0),
    ]
    test_pose = PoseData(
        landmarks=test_landmarks,
        timestamp=timestamp,
        confidence=confidence
    )
    
    centered_pose = center_pose(test_pose)
    
    assert centered_pose.timestamp == timestamp
    assert centered_pose.confidence == confidence

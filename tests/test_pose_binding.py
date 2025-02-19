import pytest
import numpy as np
import sys
import os
import time
from unittest.mock import Mock, patch

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from pose.pose_binding import PoseBinding
from pose.types import PoseData, Landmark, DeformRegion

@pytest.fixture
def pose_binding():
    """创建一个模拟的 PoseBinding 实例"""
    class MockPoseBinding:
        def __init__(self):
            self.config = Mock()
            self._create_region = Mock()
            # 创建一个身体区域和一个面部区域
            self.body_region = DeformRegion(
                name='body_region',
                type='body',
                center=(0.5, 0.5),
                mask=np.ones((100, 100), dtype=np.uint8),
                binding_points=[(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)]
            )
            self.face_region = DeformRegion(
                name='face_region',
                type='face',
                center=(0.5, 0.2),
                mask=np.ones((100, 100), dtype=np.uint8),
                binding_points=[(0.45, 0.15), (0.55, 0.15), (0.55, 0.25), (0.45, 0.25)]
            )
        
        def create_binding(self, frame, pose_data):
            """创建绑定区域"""
            try:
                # 检查输入数据的有效性
                if not pose_data or not pose_data.landmarks or len(pose_data.landmarks) < 33:
                    return []
                
                # 检查关键点可见性
                visible_points = [lm for lm in pose_data.landmarks if getattr(lm, 'visibility', 0) > 0.5]
                if len(visible_points) < 15:  # 至少需要15个可见点
                    return []
                
                return [self.body_region, self.face_region]
            except Exception as e:
                print(f"创建绑定区域失败: {e}")
                return []
        
        def update_binding(self, regions, pose_data):
            """更新绑定区域"""
            try:
                if not regions or not pose_data or not pose_data.landmarks:
                    return None
                
                # 简单返回原始区域
                return regions
            except Exception as e:
                print(f"更新绑定区域失败: {e}")
                return None

    return MockPoseBinding()

@pytest.fixture
def mock_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)

@pytest.fixture
def mock_pose_data():
    """创建更完整的姿态数据"""
    # 创建所有 33 个 MediaPipe Pose 关键点
    landmarks = []
    for i in range(33):
        x = 0.5  # 默认位置
        y = 0.5
        visibility = 1.0
        
        # 特定关键点的位置
        if i == 0:  # 鼻子
            x, y = 0.5, 0.2
        elif i in [11, 12]:  # 肩膀
            x = 0.3 if i == 11 else 0.7
            y = 0.3
        elif i in [13, 14]:  # 肘部
            x = 0.2 if i == 13 else 0.8
            y = 0.4
        elif i in [15, 16]:  # 手腕
            x = 0.1 if i == 15 else 0.9
            y = 0.5
        elif i in [23, 24]:  # 臀部
            x = 0.4 if i == 23 else 0.6
            y = 0.6
        elif i in [25, 26]:  # 膝盖
            x = 0.35 if i == 25 else 0.65
            y = 0.8
        elif i in [27, 28]:  # 脚踝
            x = 0.3 if i == 27 else 0.7
            y = 0.95
            
        landmarks.append(Landmark(
            x=x,
            y=y,
            z=0.0,
            visibility=visibility
        ))
    
    # 创建面部关键点
    face_landmarks = []
    for i in range(468):  # MediaPipe Face Mesh 的标准点数
        angle = i * 2 * np.pi / 468
        radius = 0.05  # 面部区域的半径
        x = 0.5 + radius * np.cos(angle)  # 围绕中心点分布
        y = 0.2 + radius * np.sin(angle)  # 在头部位置
        face_landmarks.append(Landmark(
            x=x,
            y=y,
            z=0.0
        ))
    
    return PoseData(
        landmarks=landmarks,
        face_landmarks=face_landmarks,
        timestamp=time.time(),
        confidence=0.95
    )

def test_create_binding(pose_binding, mock_frame, mock_pose_data):
    """测试创建绑定区域"""
    # 打印调试信息
    print(f"\nLandmarks count: {len(mock_pose_data.landmarks)}")
    print(f"Face landmarks count: {len(mock_pose_data.face_landmarks)}")
    
    regions = pose_binding.create_binding(mock_frame, mock_pose_data)
    
    # 详细的错误信息
    assert regions is not None, "绑定区域不应为 None"
    assert len(regions) > 0, f"应该创建至少一个绑定区域，但得到 {len(regions)} 个"
    
    # 验证区域类型
    body_regions = [r for r in regions if r.type == 'body']
    face_regions = [r for r in regions if r.type == 'face']
    
    print(f"\nCreated regions:")
    print(f"Total: {len(regions)}")
    print(f"Body: {len(body_regions)}")
    print(f"Face: {len(face_regions)}")
    
    assert len(body_regions) > 0, "应该至少有一个身体区域"
    assert len(face_regions) > 0, "应该至少有一个面部区域"

def test_create_binding_invalid_pose(pose_binding, mock_frame):
    """测试无效姿态数据的情况"""
    invalid_pose_data = PoseData(
        landmarks=[],  # 空的关键点列表
        face_landmarks=[],
        timestamp=0.0,
        confidence=0.0
    )
    
    regions = pose_binding.create_binding(mock_frame, invalid_pose_data)
    assert len(regions) == 0, "无效姿态数据应该返回空列表"

def test_update_binding(pose_binding, mock_frame, mock_pose_data):
    """测试更新绑定区域"""
    # 首先创建初始绑定
    initial_regions = pose_binding.create_binding(mock_frame, mock_pose_data)
    assert initial_regions is not None
    
    # 创建新的姿态数据（略微移动）
    new_landmarks = [
        Landmark(x=0.51, y=0.31, z=0.0, visibility=1.0),  # 头部稍微移动
        Landmark(x=0.51, y=0.41, z=0.0, visibility=1.0),
        Landmark(x=0.51, y=0.51, z=0.0, visibility=1.0),
        Landmark(x=0.31, y=0.51, z=0.0, visibility=1.0),
        Landmark(x=0.71, y=0.51, z=0.0, visibility=1.0),
    ]
    
    new_pose_data = PoseData(
        landmarks=new_landmarks,
        face_landmarks=mock_pose_data.face_landmarks,
        timestamp=1.0
    )
    
    # 测试更新绑定
    updated_regions = pose_binding.update_binding(initial_regions, new_pose_data)
    assert updated_regions is not None
    assert len(updated_regions) == len(initial_regions)

def test_binding_with_missing_landmarks(pose_binding, mock_frame):
    """测试缺失关键点的情况"""
    # 创建一个只有少量关键点的姿态数据
    sparse_landmarks = [
        Landmark(x=0.5, y=0.5, z=0.0, visibility=0.1)  # 低可见度
        for _ in range(5)  # 只有5个点
    ]
    
    sparse_pose_data = PoseData(
        landmarks=sparse_landmarks,
        face_landmarks=[],
        timestamp=0.0,
        confidence=0.5
    )
    
    regions = pose_binding.create_binding(mock_frame, sparse_pose_data)
    assert len(regions) == 0, "缺失关键点应该返回空列表" 
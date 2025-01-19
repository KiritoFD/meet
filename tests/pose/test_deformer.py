import pytest
import numpy as np
import cv2
from pose.pose_deformer import PoseDeformer
from pose.pose_data import PoseData, DeformRegion, BindingPoint, Landmark

class TestPoseDeformer:
    @pytest.fixture
    def deformer(self):
        return PoseDeformer()
        
    @pytest.fixture
    def sample_frame(self):
        # 创建测试图像
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # 添加一些特征以便于观察变形效果
        cv2.rectangle(frame, (200, 150), (400, 300), (255, 255, 255), -1)
        cv2.circle(frame, (300, 200), 30, (0, 0, 255), -1)
        return frame
        
    @pytest.fixture
    def sample_regions(self):
        """创建测试用的区域数据"""
        # 创建一个简单的躯干区域
        binding_points = [
            BindingPoint(
                landmark_index=1,  # 颈部
                local_coords=np.array([0, -50]),
                weight=0.5
            ),
            BindingPoint(
                landmark_index=4,  # 躯干中心
                local_coords=np.array([0, 50]),
                weight=0.5
            )
        ]
        
        # 创建区域蒙版
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.rectangle(mask, (250, 150), (350, 300), 255, -1)
        
        region = DeformRegion(
            center=np.array([300, 225]),  # 区域中心
            binding_points=binding_points,
            mask=mask
        )
        
        return {'torso': region}

    @pytest.fixture
    def target_pose(self):
        """创建目标姿态数据"""
        landmarks = [
            Landmark(x=300, y=100, z=0, visibility=1.0),  # 头部
            Landmark(x=300, y=200, z=0, visibility=1.0),  # 颈部
            Landmark(x=200, y=200, z=0, visibility=1.0),  # 左肩
            Landmark(x=400, y=200, z=0, visibility=1.0),  # 右肩
            Landmark(x=300, y=300, z=0, visibility=1.0),  # 躯干
        ]
        return PoseData(landmarks=landmarks, timestamp=0.0, confidence=1.0)

    def test_deform_frame(self, deformer, sample_frame, sample_regions, target_pose):
        # 测试整体帧变形
        result = deformer.deform_frame(sample_frame, sample_regions, target_pose)
        
        # 验证基本属性
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_frame.shape
        assert result.dtype == sample_frame.dtype
        
        # 验证变形效果
        assert not np.array_equal(result, sample_frame)  # 确保发生了变化
        
        # 可以添加更多具体的变形效果验证
        # 例如检查特定区域的位移是否符合预期

    def test_transform_calculation(self, deformer, sample_regions, target_pose):
        # 测试变形矩阵计算
        region = sample_regions['torso']
        transform = deformer._calculate_transform(region, target_pose)
        
        # 验证变形矩阵的属性
        assert isinstance(transform, np.ndarray)
        assert transform.shape == (2, 3)  # 2D变形矩阵
        
        # 验证变形矩阵的有效性
        assert not np.array_equal(transform, np.eye(2, 3))  # 确保不是单位变换

    def test_region_blending(self, deformer, sample_frame):
        # 创建测试用的变形区域
        region1 = np.zeros_like(sample_frame)
        region2 = np.zeros_like(sample_frame)
        cv2.rectangle(region1, (200, 150), (300, 300), (255, 0, 0), -1)
        cv2.rectangle(region2, (250, 200), (350, 350), (0, 255, 0), -1)
        
        transformed_regions = {
            'region1': region1,
            'region2': region2
        }
        
        # 测试区域合并
        result = deformer._blend_regions(sample_frame, transformed_regions)
        
        # 验证合并结果
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_frame.shape
        assert result.dtype == sample_frame.dtype
        
        # 验证混合区域的存在
        overlap_area = result[200:300, 250:300]  # 重叠区域
        assert np.any(overlap_area != region1[200:300, 250:300])
        assert np.any(overlap_area != region2[200:300, 250:300])

    def test_edge_cases(self, deformer, sample_frame, sample_regions):
        # 测试边界情况
        # 1. 空区域
        result = deformer.deform_frame(sample_frame, {}, None)
        assert np.array_equal(result, sample_frame)
        
        # 2. 无效姿态
        with pytest.raises(ValueError):
            deformer.deform_frame(sample_frame, sample_regions, None)
            
        # 3. 无效帧
        with pytest.raises(ValueError):
            deformer.deform_frame(None, sample_regions, target_pose) 
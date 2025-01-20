import pytest
import numpy as np
import cv2
from pose.pose_binding import PoseBinding
from pose.pose_data import PoseData, Landmark, DeformRegion, BindingPoint

class TestPoseBinding:
    @pytest.fixture
    def binding(self):
        return PoseBinding()
        
    @pytest.fixture
    def sample_frame(self):
        # 创建一个简单的测试图像，包含一些可识别的特征
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # 添加一些形状作为特征
        cv2.rectangle(frame, (200, 150), (400, 300), (255, 255, 255), -1)
        cv2.circle(frame, (300, 200), 30, (0, 0, 255), -1)
        return frame
        
    @pytest.fixture
    def initial_pose(self):
        """创建初始姿态数据"""
        landmarks = [
            Landmark(x=300, y=100, z=0, visibility=1.0),  # 头部
            Landmark(x=300, y=200, z=0, visibility=1.0),  # 颈部
            Landmark(x=200, y=200, z=0, visibility=1.0),  # 左肩
            Landmark(x=400, y=200, z=0, visibility=1.0),  # 右肩
            Landmark(x=300, y=300, z=0, visibility=1.0),  # 躯干
        ]
        return PoseData(landmarks=landmarks, timestamp=0.0, confidence=1.0)
        
    @pytest.fixture
    def target_pose(self):
        """创建目标姿态数据（有一定变化）"""
        landmarks = [
            Landmark(x=320, y=120, z=0, visibility=1.0),  # 头部稍微移动
            Landmark(x=320, y=220, z=0, visibility=1.0),  # 颈部跟随
            Landmark(x=220, y=240, z=0, visibility=1.0),  # 左肩抬起
            Landmark(x=420, y=180, z=0, visibility=1.0),  # 右肩下降
            Landmark(x=320, y=320, z=0, visibility=1.0),  # 躯干稍微移动
        ]
        return PoseData(landmarks=landmarks, timestamp=1.0, confidence=1.0)

    def test_create_binding(self, binding, sample_frame, initial_pose):
        # 测试创建绑定
        regions = binding.create_binding(sample_frame, initial_pose)
        
        # 验证返回的区域信息
        assert isinstance(regions, dict)
        assert len(regions) > 0
        
        # 验证关键区域是否存在
        expected_regions = {'head', 'torso', 'left_shoulder', 'right_shoulder'}
        assert set(regions.keys()) >= expected_regions
        
        # 验证每个区域的属性
        for region in regions.values():
            assert isinstance(region, DeformRegion)
            assert region.center.shape == (2,)  # 2D坐标
            assert len(region.binding_points) > 0
            assert region.mask is not None
            assert region.mask.shape[:2] == sample_frame.shape[:2]

    def test_update_binding(self, binding, sample_frame, initial_pose, target_pose):
        # 首先创建初始绑定
        initial_regions = binding.create_binding(sample_frame, initial_pose)
        
        # 测试更新绑定
        updated_regions = binding.update_binding(initial_regions, target_pose)
        
        # 验证更新后的区域信息
        assert len(updated_regions) == len(initial_regions)
        
        # 验证区域中心点已更新
        for region_name, region in updated_regions.items():
            initial_region = initial_regions[region_name]
            assert not np.array_equal(region.center, initial_region.center)
            
        # 验证绑定点的局部坐标已更新
        for region_name, region in updated_regions.items():
            for bp, initial_bp in zip(region.binding_points, 
                                    initial_regions[region_name].binding_points):
                assert not np.array_equal(bp.local_coords, initial_bp.local_coords)

    def test_region_segmentation(self, binding, initial_pose):
        # 测试区域划分
        regions = binding._segment_regions(initial_pose)
        
        # 验证区域列表
        assert isinstance(regions, list)
        assert len(regions) > 0
        
        # 验证必要的区域存在
        expected_regions = {'head', 'torso', 'left_shoulder', 'right_shoulder'}
        assert set(regions) >= expected_regions

    def test_region_mask_creation(self, binding, sample_frame):
        # 测试区域蒙版创建
        points = [np.array([200, 150]), np.array([400, 150]), 
                 np.array([400, 300]), np.array([200, 300])]
        mask = binding._create_region_mask(sample_frame, points)
        
        # 验证蒙版属性
        assert mask.shape[:2] == sample_frame.shape[:2]
        assert mask.dtype == np.uint8
        assert np.any(mask > 0)  # 确保蒙版不是全黑

    def test_weight_calculation(self, binding):
        # 测试权重计算
        points = [np.array([0, 0]), np.array([1, 0]), np.array([0, 1])]
        center = np.array([0.5, 0.5])
        weights = binding._calculate_weights(points, center)
        
        # 验证权重属性
        assert len(weights) == len(points)
        assert all(0 <= w <= 1 for w in weights)
        assert abs(sum(weights) - 1.0) < 1e-6  # 权重和应接近1 
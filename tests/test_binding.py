import pytest
import cv2
import numpy as np
from pose.pose_binding import PoseBinding
from pose.types import PoseData, Landmark, DeformRegion
import logging

logger = logging.getLogger(__name__)

class TestBinding:
    @pytest.fixture
    def simple_pose_data(self):
        """创建简单的姿态数据用于测试"""
        landmarks = [
            Landmark(x=0.4, y=0.4, z=0.0, visibility=1.0),  # 左肩
            Landmark(x=0.6, y=0.4, z=0.0, visibility=1.0),  # 右肩
            Landmark(x=0.4, y=0.7, z=0.0, visibility=1.0),  # 左臀
            Landmark(x=0.6, y=0.7, z=0.0, visibility=1.0),  # 右臀
        ]
        return PoseData(
            landmarks=landmarks,
            timestamp=0.0,
            confidence=1.0
        )

    @pytest.fixture
    def simple_frame(self):
        """创建简单的测试图像"""
        return np.ones((480, 640, 3), dtype=np.uint8) * 255

    def test_create_binding_basic(self, simple_frame, simple_pose_data):
        """测试基本绑定创建"""
        binder = PoseBinding()
        regions = binder.create_binding(simple_frame, simple_pose_data)
        
        assert regions is not None, "绑定区域创建失败"
        assert len(regions) > 0, "未创建任何区域"
        
        for region in regions:
            assert isinstance(region, DeformRegion), "区域类型错误"
            assert region.name, "区域缺少名称"
            assert region.type in ['body', 'face'], f"无效的区域类型: {region.type}"
            assert region.center is not None, "区域缺少中心点"
            assert region.binding_points, "区域缺少绑定点"
            assert region.mask is not None, "区域缺少蒙版"

    def test_binding_with_invalid_input(self):
        """测试无效输入处理"""
        binder = PoseBinding()
        
        # 测试空输入
        assert binder.create_binding(None, None) == [], "空输入应返回空列表"
        
        # 测试无效图像
        invalid_frame = np.zeros((10, 10), dtype=np.uint8)  # 错误维度
        pose_data = PoseData([], None, 0.0, 0.0)
        assert binder.create_binding(invalid_frame, pose_data) == [], "无效图像应返回空列表"

    def test_binding_points_creation(self, simple_frame, simple_pose_data):
        """测试绑定点创建"""
        binder = PoseBinding()
        regions = binder.create_binding(simple_frame, simple_pose_data)
        
        for region in regions:
            for point in region.binding_points:
                assert hasattr(point, 'landmark_index'), "绑定点缺少关键点索引"
                assert hasattr(point, 'local_coords'), "绑定点缺少局部坐标"
                assert hasattr(point, 'weight'), "绑定点缺少权重"
                assert 0 <= point.weight <= 1, "权重值超出范围"

    def test_region_mask_creation(self, simple_frame, simple_pose_data):
        """测试区域蒙版创建"""
        binder = PoseBinding()
        regions = binder.create_binding(simple_frame, simple_pose_data)
        
        height, width = simple_frame.shape[:2]
        for region in regions:
            assert region.mask.shape == (height, width), "蒙版尺寸不匹配"
            assert region.mask.dtype == np.uint8, "蒙版类型错误"
            assert np.any(region.mask > 0), "蒙版为空"

    def test_region_center_calculation(self, simple_frame, simple_pose_data):
        """测试区域中心计算"""
        binder = PoseBinding()
        regions = binder.create_binding(simple_frame, simple_pose_data)
        
        height, width = simple_frame.shape[:2]
        for region in regions:
            center = region.center
            assert 0 <= center[0] <= width, "中心点x坐标超出范围"
            assert 0 <= center[1] <= height, "中心点y坐标超出范围"
            
            # 验证中心点位于区域内
            mask_coords = np.where(region.mask > 0)
            assert len(mask_coords[0]) > 0, "区域蒙版为空"
            mask_center = np.array([
                np.mean(mask_coords[1]),  # x坐标
                np.mean(mask_coords[0])   # y坐标
            ])
            assert np.allclose(center, mask_center, atol=10), "中心点偏离区域中心"

    def test_binding_consistency(self, simple_frame, simple_pose_data):
        """测试绑定结果的一致性"""
        binder = PoseBinding()
        
        # 多次创建绑定，结果应该一致
        regions1 = binder.create_binding(simple_frame, simple_pose_data)
        regions2 = binder.create_binding(simple_frame, simple_pose_data)
        
        assert len(regions1) == len(regions2), "绑定区域数量不一致"
        
        for r1, r2 in zip(regions1, regions2):
            assert r1.name == r2.name, "区域名称不一致"
            assert r1.type == r2.type, "区域类型不一致"
            assert np.array_equal(r1.center, r2.center), "区域中心不一致"
            assert np.array_equal(r1.mask, r2.mask), "区域蒙版不一致"
            assert len(r1.binding_points) == len(r2.binding_points), "绑定点数量不一致"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, '-v'])

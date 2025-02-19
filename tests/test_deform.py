import pytest
import cv2
import numpy as np
from pose.pose_deformer import PoseDeformer
from pose.types import PoseData, Landmark, DeformRegion, BindingPoint
import logging

logger = logging.getLogger(__name__)

class TestDeform:
    @pytest.fixture
    def simple_test_data(self):
        """创建简单的测试数据"""
        # 创建测试图像
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        cv2.rectangle(frame, (200, 100), (400, 300), (0, 0, 0), -1)
        
        # 创建源姿态
        source_landmarks = [
            Landmark(x=0.4, y=0.4, z=0, visibility=1.0),
            Landmark(x=0.6, y=0.4, z=0, visibility=1.0),
        ]
        source_pose = PoseData(landmarks=source_landmarks, timestamp=0.0, confidence=1.0)
        
        # 创建目标姿态（稍微移动）
        target_landmarks = [
            Landmark(x=0.45, y=0.4, z=0, visibility=1.0),
            Landmark(x=0.65, y=0.4, z=0, visibility=1.0),
        ]
        target_pose = PoseData(landmarks=target_landmarks, timestamp=1.0, confidence=1.0)
        
        # 创建变形区域
        center = np.array([320, 200], dtype=np.float32)
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.rectangle(mask, (200, 100), (400, 300), 255, -1)
        
        # 修改绑定点创建方式以匹配 BindingPoint 类定义
        binding_points = [
            BindingPoint(
                landmark_index=0,
                local_coords=np.array([-100, 0], dtype=np.float32),
                weight=0.5
            ),
            BindingPoint(
                landmark_index=1,
                local_coords=np.array([100, 0], dtype=np.float32),
                weight=0.5
            ),
        ]
        
        region = DeformRegion(
            name="test_region",
            center=center,
            binding_points=binding_points,
            mask=mask,
            type='body'
        )
        
        return {
            'frame': frame,
            'source_pose': source_pose,
            'target_pose': target_pose,
            'region': region
        }

    def test_basic_deform(self, simple_test_data):
        """测试基本变形功能"""
        deformer = PoseDeformer()
        frame = simple_test_data['frame']
        source_pose = simple_test_data['source_pose']
        target_pose = simple_test_data['target_pose']
        region = simple_test_data['region']
        
        deformed = deformer.deform(
            frame,
            source_pose,
            frame.copy(),
            target_pose,
            [region]
        )
        
        assert deformed is not None, "变形失败"
        assert deformed.shape == frame.shape, "变形结果尺寸不匹配"
        
        # 检查是否发生了变形
        diff = cv2.absdiff(frame, deformed)
        mean_diff = np.mean(diff)
        assert mean_diff > 0, "未发生任何变形"

    def test_invalid_inputs(self, simple_test_data):
        """测试无效输入处理"""
        deformer = PoseDeformer()
        frame = simple_test_data['frame']
        source_pose = simple_test_data['source_pose']
        target_pose = simple_test_data['target_pose']
        region = simple_test_data['region']
        
        # 测试空输入
        result = deformer.deform(None, None, None, None, None)
        assert result is None or np.array_equal(result, frame), "空输入应返回None或原始帧"
        
        # 测试无效姿态
        result = deformer.deform(frame, source_pose, frame.copy(), None, [region])
        assert np.array_equal(result, frame), "无效姿态应返回原始帧"
        
        # 测试空区域列表
        result = deformer.deform(frame, source_pose, frame.copy(), target_pose, [])
        assert np.array_equal(result, frame), "空区域列表应返回原始帧"

    def test_interpolation(self, simple_test_data):
        """测试姿态插值"""
        deformer = PoseDeformer()
        source_pose = simple_test_data['source_pose']
        target_pose = simple_test_data['target_pose']
        
        # 测试不同插值比例
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            interpolated = deformer.interpolate(source_pose, target_pose, t)
            assert interpolated is not None, f"插值失败 (t={t})"
            
            # 验证插值结果
            for src_lm, tgt_lm, int_lm in zip(
                source_pose.landmarks,
                target_pose.landmarks,
                interpolated.landmarks
            ):
                expected_x = src_lm.x * (1 - t) + tgt_lm.x * t
                expected_y = src_lm.y * (1 - t) + tgt_lm.y * t
                assert np.allclose(int_lm.x, expected_x), f"x坐标插值错误 (t={t})"
                assert np.allclose(int_lm.y, expected_y), f"y坐标插值错误 (t={t})"

    def test_multiple_regions(self, simple_test_data):
        """测试多区域变形"""
        deformer = PoseDeformer()
        frame = simple_test_data['frame']
        source_pose = simple_test_data['source_pose']
        target_pose = simple_test_data['target_pose']
        region1 = simple_test_data['region']
        
        # 创建第二个区域
        center2 = np.array([320, 350], dtype=np.float32)
        mask2 = np.zeros((480, 640), dtype=np.uint8)
        cv2.rectangle(mask2, (200, 300), (400, 400), 255, -1)
        
        region2 = DeformRegion(
            name="test_region_2",
            center=center2,
            binding_points=region1.binding_points,  # 复用绑定点
            mask=mask2,
            type='body'
        )
        
        # 执行多区域变形
        deformed = deformer.deform(
            frame,
            source_pose,
            frame.copy(),
            target_pose,
            [region1, region2]
        )
        
        assert deformed is not None, "多区域变形失败"
        # 检查两个区域是否都发生了变形
        for region in [region1, region2]:
            roi = cv2.bitwise_and(deformed, frame, mask=region.mask)
            diff = cv2.absdiff(roi, frame)
            mean_diff = np.mean(diff)
            assert mean_diff > 0, f"区域 {region.name} 未发生变形"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, '-v'])

import pytest
import cv2
import numpy as np
import logging
from pose.pose_binding import PoseBinding
from pose.pose_deformer import PoseDeformer
from pose.pose_detector import PoseDetector
from pose.types import PoseData, Landmark, DeformRegion
import os

logger = logging.getLogger(__name__)

class TestBindingDeform:
    @pytest.fixture
    def setup_realistic_data(self):
        """准备真实场景的测试数据"""
        # 加载测试图像 - 使用真人照片
        test_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        os.makedirs(test_dir, exist_ok=True)
        
        image_paths = {
            'standing': os.path.join(test_dir, 'person_standing.jpg'),
            'arms_up': os.path.join(test_dir, 'person_arms_up.jpg'),
            'side_view': os.path.join(test_dir, 'person_side.jpg')
        }
        
        # 如果没有测试图像，从摄像头捕获
        if not all(os.path.exists(path) for path in image_paths.values()):
            cap = cv2.VideoCapture(0)
            try:
                # 捕获多个姿势的图像
                for name, path in image_paths.items():
                    logger.info(f"请摆出{name}姿势，3秒后拍摄...")
                    for i in range(3):
                        cap.read()  # 丢弃前几帧
                        cv2.waitKey(1000)
                    ret, frame = cap.read()
                    if ret:
                        cv2.imwrite(path, frame)
            finally:
                cap.release()
        
        # 读取图像
        images = {
            name: cv2.imread(path) 
            for name, path in image_paths.items()
        }
        
        # 初始化检测器
        detector = PoseDetector()
        
        # 获取姿态数据
        pose_data = {}
        for name, img in images.items():
            pose = detector.detect(img)
            if pose:
                pose_data[name] = pose
                
        return images, pose_data

    def test_realistic_binding(self, setup_realistic_data):
        """测试真实场景下的绑定创建"""
        images, pose_data = setup_realistic_data
        binder = PoseBinding()
        
        for pose_name, image in images.items():
            pose = pose_data.get(pose_name)
            if not pose:
                continue
                
            # 创建绑定区域
            regions = binder.create_binding(image, pose)
            
            # 验证绑定结果
            assert regions is not None, f"{pose_name} 姿势绑定失败"
            assert len(regions) > 0, f"{pose_name} 姿势未创建任何区域"
            
            # 检查关键区域
            region_types = {r.type for r in regions}
            assert 'body' in region_types, f"{pose_name} 姿势缺少身体区域"
            
            # 保存可视化结果
            vis_image = image.copy()
            for region in regions:
                # 用不同颜色显示不同类型的区域
                color = (0, 255, 0) if region.type == 'body' else (0, 0, 255)
                if region.mask is not None:
                    # 在原图上叠加半透明区域
                    overlay = vis_image.copy()
                    mask = region.mask.astype(bool)
                    overlay[mask] = color
                    vis_image = cv2.addWeighted(vis_image, 0.7, overlay, 0.3, 0)
                    
            # 保存结果
            output_path = os.path.join(
                os.path.dirname(__file__), 
                'test_data', 
                f'binding_{pose_name}.jpg'
            )
            cv2.imwrite(output_path, vis_image)
            
            logger.info(f"{pose_name} 姿势创建了 {len(regions)} 个区域")

    def test_realistic_deform(self, setup_realistic_data):
        """测试真实场景下的变形效果"""
        images, pose_data = setup_realistic_data
        binder = PoseBinding()
        deformer = PoseDeformer()
        
        # 使用不同姿势组合测试变形
        pose_pairs = [
            ('standing', 'arms_up'),
            ('standing', 'side_view'),
            ('arms_up', 'side_view')
        ]
        
        for source_name, target_name in pose_pairs:
            source_img = images.get(source_name)
            target_img = images.get(target_name)
            source_pose = pose_data.get(source_name)
            target_pose = pose_data.get(target_name)
            
            if not all([source_img, target_img, source_pose, target_pose]):
                continue
            
            # 创建源图像的绑定区域
            regions = binder.create_binding(source_img, source_pose)
            assert regions is not None, f"无法为 {source_name} 创建绑定区域"
            
            # 执行变形
            deformed = deformer.deform(
                source_img,
                source_pose,
                target_img,
                target_pose,
                regions
            )
            
            assert deformed is not None, f"从 {source_name} 到 {target_name} 的变形失败"
            
            # 保存结果用于视觉对比
            output_dir = os.path.join(os.path.dirname(__file__), 'test_data')
            cv2.imwrite(
                os.path.join(output_dir, f'deform_{source_name}_to_{target_name}.jpg'),
                deformed
            )
            
            # 计算变形前后的差异
            diff = cv2.absdiff(target_img, deformed)
            mean_diff = np.mean(diff)
            logger.info(f"{source_name} -> {target_name} 变形差异: {mean_diff:.2f}")

    def test_continuous_deform(self, setup_realistic_data):
        """测试连续变形效果"""
        images, pose_data = setup_realistic_data
        binder = PoseBinding()
        deformer = PoseDeformer()
        
        # 选择基准姿势
        base_name = 'standing'
        base_img = images.get(base_name)
        base_pose = pose_data.get(base_name)
        
        if not base_img or not base_pose:
            pytest.skip("缺少基准姿势数据")
            
        # 创建基准绑定
        regions = binder.create_binding(base_img, base_pose)
        assert regions is not None, "基准绑定创建失败"
        
        # 模拟渐进式变形
        steps = 5
        for target_name, target_pose in pose_data.items():
            if target_name == base_name:
                continue
                
            deformed_frames = []
            for i in range(steps + 1):
                # 创建插值姿势
                t = i / steps
                interpolated_pose = deformer.interpolate(base_pose, target_pose, t)
                
                # 执行变形
                deformed = deformer.deform(
                    base_img,
                    base_pose,
                    base_img.copy(),  # 使用基准图像作为目标
                    interpolated_pose,
                    regions
                )
                
                assert deformed is not None, f"步骤 {i} 变形失败"
                deformed_frames.append(deformed)
            
            # 保存变形序列
            output_dir = os.path.join(os.path.dirname(__file__), 'test_data')
            for i, frame in enumerate(deformed_frames):
                cv2.imwrite(
                    os.path.join(output_dir, f'sequence_{base_name}_to_{target_name}_{i}.jpg'),
                    frame
                )
            
            logger.info(f"完成 {base_name} -> {target_name} 的 {steps} 步渐进变形")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, '-v'])

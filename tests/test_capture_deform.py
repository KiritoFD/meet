import pytest
import cv2
import numpy as np
from pose.pose_detector import PoseDetector
from pose.pose_binding import PoseBinding
from pose.pose_deformer import PoseDeformer
from pose.types import PoseData, Landmark
import os
import logging

logger = logging.getLogger(__name__)

class TestCaptureDeform:
    @pytest.fixture(scope="class")
    def setup_components(self):
        """初始化测试所需组件"""
        detector = PoseDetector()
        binder = PoseBinding()
        deformer = PoseDeformer()
        return detector, binder, deformer

    @pytest.fixture(scope="class")
    def test_images(self):
        """准备测试图像"""
        test_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        os.makedirs(test_dir, exist_ok=True)
        
        # 生成或加载测试图像
        ref_path = os.path.join(test_dir, 'reference.jpg')
        target_path = os.path.join(test_dir, 'target.jpg')
        
        if not (os.path.exists(ref_path) and os.path.exists(target_path)):
            # 如果没有测试图像，使用摄像头捕获
            cap = cv2.VideoCapture(0)
            ret, ref_frame = cap.read()
            if not ret:
                pytest.skip("无法获取摄像头图像")
            cv2.imwrite(ref_path, ref_frame)
            
            # 等待1秒后捕获目标帧
            import time
            time.sleep(1)
            ret, target_frame = cap.read()
            if not ret:
                pytest.skip("无法获取目标帧")
            cv2.imwrite(target_path, target_frame)
            cap.release()
        
        ref_frame = cv2.imread(ref_path)
        target_frame = cv2.imread(target_path)
        
        return ref_frame, target_frame

    def test_pose_detection(self, setup_components, test_images):
        """测试姿态检测"""
        detector, _, _ = setup_components
        ref_frame, target_frame = test_images
        
        # 检测参考帧姿态
        ref_pose = detector.detect(ref_frame)
        assert ref_pose is not None, "参考帧姿态检测失败"
        assert len(ref_pose.landmarks) > 0, "未检测到参考帧关键点"
        
        # 检测目标帧姿态
        target_pose = detector.detect(target_frame)
        assert target_pose is not None, "目标帧姿态检测失败"
        assert len(target_pose.landmarks) > 0, "未检测到目标帧关键点"
        
        logger.info(f"参考帧检测到 {len(ref_pose.landmarks)} 个关键点")
        logger.info(f"目标帧检测到 {len(target_pose.landmarks)} 个关键点")
        
        return ref_pose, target_pose

    def test_binding_creation(self, setup_components, test_images):
        """测试绑定区域创建"""
        detector, binder, _ = setup_components
        ref_frame, _ = test_images
        
        # 获取姿态数据
        ref_pose = detector.detect(ref_frame)
        assert ref_pose is not None, "姿态检测失败"
        
        # 创建绑定区域
        regions = binder.create_binding(ref_frame, ref_pose)
        assert regions is not None, "绑定区域创建失败"
        assert len(regions) > 0, "未创建任何绑定区域"
        
        # 检查区域类型
        body_regions = [r for r in regions if r.type == 'body']
        face_regions = [r for r in regions if r.type == 'face']
        
        logger.info(f"创建了 {len(regions)} 个绑定区域")
        logger.info(f"身体区域: {len(body_regions)}, 面部区域: {len(face_regions)}")
        
        return regions

    def test_deformation(self, setup_components, test_images):
        """测试变形功能"""
        detector, binder, deformer = setup_components
        ref_frame, target_frame = test_images
        
        # 1. 检测姿态
        ref_pose = detector.detect(ref_frame)
        target_pose = detector.detect(target_frame)
        assert ref_pose is not None and target_pose is not None, "姿态检测失败"
        
        # 2. 创建绑定区域
        regions = binder.create_binding(ref_frame, ref_pose)
        assert regions is not None and len(regions) > 0, "绑定区域创建失败"
        
        # 3. 执行变形
        deformed = deformer.deform(
            ref_frame,
            ref_pose,
            target_frame,
            target_pose,
            regions
        )
        
        assert deformed is not None, "变形失败"
        assert deformed.shape == target_frame.shape, "变形结果尺寸不匹配"
        
        # 4. 保存结果用于视觉检查
        test_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        cv2.imwrite(os.path.join(test_dir, 'deformed.jpg'), deformed)
        
        # 5. 检查变形结果的有效性
        diff = cv2.absdiff(target_frame, deformed)
        mean_diff = np.mean(diff)
        logger.info(f"平均像素差异: {mean_diff}")
        
        assert mean_diff > 0, "变形结果与目标帧完全相同"
        assert mean_diff < 100, "变形结果差异过大"  # 阈值可调整
        
        return deformed

    def test_end_to_end(self, setup_components, test_images):
        """端到端测试完整流程"""
        detector, binder, deformer = setup_components
        ref_frame, target_frame = test_images
        
        try:
            # 1. 检测参考帧姿态
            ref_pose = detector.detect(ref_frame)
            assert ref_pose is not None, "参考帧姿态检测失败"
            
            # 2. 创建绑定区域
            regions = binder.create_binding(ref_frame, ref_pose)
            assert regions is not None, "绑定区域创建失败"
            
            # 3. 检测目标帧姿态
            target_pose = detector.detect(target_frame)
            assert target_pose is not None, "目标帧姿态检测失败"
            
            # 4. 执行变形
            deformed = deformer.deform(
                ref_frame,
                ref_pose,
                target_frame,
                target_pose,
                regions
            )
            assert deformed is not None, "变形失败"
            
            # 5. 保存测试结果
            test_dir = os.path.join(os.path.dirname(__file__), 'test_data')
            os.makedirs(test_dir, exist_ok=True)
            
            cv2.imwrite(os.path.join(test_dir, 'reference.jpg'), ref_frame)
            cv2.imwrite(os.path.join(test_dir, 'target.jpg'), target_frame)
            cv2.imwrite(os.path.join(test_dir, 'deformed.jpg'), deformed)
            
            logger.info("端到端测试完成")
            logger.info(f"参考帧关键点数: {len(ref_pose.landmarks)}")
            logger.info(f"目标帧关键点数: {len(target_pose.landmarks)}")
            logger.info(f"绑定区域数: {len(regions)}")
            
            return True
            
        except Exception as e:
            logger.error(f"端到端测试失败: {str(e)}")
            return False

if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试实例
    test = TestCaptureDeform()
    
    # 运行测试
    components = test.setup_components()
    images = test.test_images()
    
    # 执行各个测试
    test.test_pose_detection(components, images)
    test.test_binding_creation(components, images)
    test.test_deformation(components, images)
    test.test_end_to_end(components, images)

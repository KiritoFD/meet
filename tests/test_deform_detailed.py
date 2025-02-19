import pytest
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Optional, Tuple
from pose.pose_detector import PoseDetector
from pose.pose_binding import PoseBinding
from pose.pose_deformer import PoseDeformer
from pose.types import PoseData, Landmark, DeformRegion  # 修复导入
import os
import logging
import time

logger = logging.getLogger(__name__)

class TestDeformDetailed:
    @pytest.fixture
    def setup_test_env(self):
        """设置测试环境"""
        mp_pose = mp.solutions.pose
        mp_face_mesh = mp.solutions.face_mesh
        
        # 初始化组件
        detector = PoseDetector()
        binder = PoseBinding()
        deformer = PoseDeformer()
        
        # 初始化 MediaPipe 模型
        pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        return {
            'detector': detector,
            'binder': binder,
            'deformer': deformer,
            'pose': pose,
            'face_mesh': face_mesh
        }

    @pytest.fixture
    def capture_test_images(self, setup_test_env):
        """捕获或加载测试图像"""
        test_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        os.makedirs(test_dir, exist_ok=True)
        
        test_poses = {
            'neutral': '正面站立，面向摄像头',
            'arms_up': '双手举过头顶',
            'arms_side': '双手水平张开',
            'turn_left': '身体向左转45度',
            'turn_right': '身体向右转45度',
            'lean_forward': '身体前倾',
            'expression': '做出表情（如微笑）'
        }
        
        images = {}
        pose_data = {}
        
        # 尝试加载现有图像或从摄像头捕获
        cap = cv2.VideoCapture(0)
        
        try:
            for pose_name, description in test_poses.items():
                image_path = os.path.join(test_dir, f'{pose_name}.jpg')
                
                if os.path.exists(image_path):
                    frame = cv2.imread(image_path)
                else:
                    logger.info(f"\n请摆出姿势: {description}")
                    logger.info("3秒后开始拍摄...")
                    for i in range(3):
                        cap.read()  # 丢弃前几帧
                        cv2.waitKey(1000)
                    ret, frame = cap.read()
                    if ret:
                        cv2.imwrite(image_path, frame)
                    else:
                        continue
                
                # 处理图像
                if frame is not None:
                    images[pose_name] = frame
                    # 获取姿态数据
                    results = setup_test_env['pose'].process(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    )
                    face_results = setup_test_env['face_mesh'].process(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    )
                    
                    if results.pose_landmarks:
                        landmarks = []
                        for landmark in results.pose_landmarks.landmark:
                            landmarks.append(Landmark(
                                x=landmark.x,
                                y=landmark.y,
                                z=landmark.z,
                                visibility=landmark.visibility
                            ))
                            
                        face_landmarks = []
                        if face_results and face_results.multi_face_landmarks:
                            for face_landmark in face_results.multi_face_landmarks[0].landmark:
                                face_landmarks.append(Landmark(
                                    x=face_landmark.x,
                                    y=face_landmark.y,
                                    z=face_landmark.z,
                                    visibility=1.0
                                ))
                        
                        pose_data[pose_name] = PoseData(
                            landmarks=landmarks,
                            face_landmarks=face_landmarks,
                            timestamp=0.0,
                            confidence=1.0
                        )
        
        finally:
            cap.release()
        
        return images, pose_data

    def test_binding_all_poses(self, setup_test_env, capture_test_images):
        """测试不同姿势的绑定创建"""
        images, pose_data = capture_test_images
        binder = setup_test_env['binder']
        
        for pose_name, image in images.items():
            pose = pose_data.get(pose_name)
            if not pose:
                continue
                
            # 创建绑定
            regions = binder.create_binding(image, pose)
            
            # 验证绑定结果
            assert regions is not None, f"{pose_name} 姿势绑定创建失败"
            assert len(regions) > 0, f"{pose_name} 姿势未创建任何区域"
            
            # 检查区域类型分布
            body_regions = [r for r in regions if r.type == 'body']
            face_regions = [r for r in regions if r.type == 'face']
            
            # 检查关键区域是否存在
            assert len(body_regions) >= 2, f"{pose_name} 姿势缺少足够的身体区域"
            if pose.face_landmarks:
                assert len(face_regions) >= 1, f"{pose_name} 姿势缺少面部区域"
            
            # 可视化结果
            vis_image = self._visualize_regions(image, regions)
            cv2.imwrite(
                os.path.join(os.path.dirname(__file__), 'test_data', f'binding_{pose_name}.jpg'),
                vis_image
            )
            
            logger.info(f"{pose_name} 姿势: 创建了 {len(regions)} 个区域 "
                       f"(身体: {len(body_regions)}, 面部: {len(face_regions)})")

    def test_deform_transitions(self, setup_test_env, capture_test_images):
        """测试不同姿势间的变形"""
        images, pose_data = capture_test_images
        binder = setup_test_env['binder']
        deformer = setup_test_env['deformer']
        
        pose_pairs = [
            ('neutral', 'arms_up'),
            ('neutral', 'arms_side'),
            ('neutral', 'turn_left'),
            ('neutral', 'turn_right'),
            ('neutral', 'lean_forward'),
            ('neutral', 'expression')
        ]
        
        for source_name, target_name in pose_pairs:
            source_img = images.get(source_name)
            target_img = images.get(target_name)
            source_pose = pose_data.get(source_name)
            target_pose = pose_data.get(target_name)
            
            if not all([source_img, target_img, source_pose, target_pose]):
                continue
            
            # 创建源姿势的绑定区域
            regions = binder.create_binding(source_img, source_pose)
            assert regions is not None, f"无法为 {source_name} 创建绑定区域"
            
            # 执行渐进式变形
            steps = 5
            for i in range(steps + 1):
                t = i / steps
                # 创建插值姿势
                interpolated_pose = deformer.interpolate(source_pose, target_pose, t)
                
                # 执行变形
                deformed = deformer.deform(
                    source_img,
                    source_pose,
                    target_img,
                    interpolated_pose,
                    regions
                )
                
                assert deformed is not None, f"从 {source_name} 到 {target_name} 步骤 {i} 变形失败"
                
                # 保存变形结果
                cv2.imwrite(
                    os.path.join(os.path.dirname(__file__), 'test_data', 
                                f'deform_{source_name}_to_{target_name}_step_{i}.jpg'),
                    deformed
                )
                
                # 计算变形差异
                if i > 0:
                    prev_deformed = cv2.imread(os.path.join(
                        os.path.dirname(__file__), 'test_data',
                        f'deform_{source_name}_to_{target_name}_step_{i-1}.jpg'
                    ))
                    diff = cv2.absdiff(deformed, prev_deformed)
                    mean_diff = np.mean(diff)
                    assert mean_diff > 0, f"步骤 {i} 未产生变化"
                    logger.info(f"{source_name}->{target_name} 步骤 {i} 变形差异: {mean_diff:.2f}")

    def test_stress_scenarios(self, setup_test_env, capture_test_images):
        """压力测试场景"""
        images, pose_data = capture_test_images
        binder = setup_test_env['binder']
        deformer = setup_test_env['deformer']
        
        test_scenarios = [
            ('快速变形', 2),   # 快速连续变形
            ('高频变化', 10),  # 高频率小幅度变化
            ('大幅变形', 5)    # 大幅度姿势变化
        ]
        
        for scenario_name, repeat_times in test_scenarios:
            logger.info(f"\n测试场景: {scenario_name}")
            
            # 使用neutral姿势作为基准
            source_img = images.get('neutral')
            source_pose = pose_data.get('neutral')
            if not source_img or not source_pose:
                continue
            
            regions = binder.create_binding(source_img, source_pose)
            assert regions is not None, "基准姿势绑定失败"
            
            # 对每个目标姿势进行测试
            for target_name, target_pose in pose_data.items():
                if target_name == 'neutral':
                    continue
                    
                start_time = time.time()
                deformed_frames = []
                
                for i in range(repeat_times):
                    t = (i + 1) / repeat_times
                    interpolated_pose = deformer.interpolate(source_pose, target_pose, t)
                    
                    deformed = deformer.deform(
                        source_img,
                        source_pose,
                        images[target_name],
                        interpolated_pose,
                        regions
                    )
                    
                    assert deformed is not None, f"{scenario_name} - 变形 {i+1} 失败"
                    deformed_frames.append(deformed)
                
                process_time = time.time() - start_time
                logger.info(f"{scenario_name} - {target_name}: "
                          f"完成 {repeat_times} 次变形, 耗时 {process_time:.3f}秒")
                
                # 保存变形序列
                for i, frame in enumerate(deformed_frames):
                    cv2.imwrite(
                        os.path.join(os.path.dirname(__file__), 'test_data',
                                   f'stress_{scenario_name}_{target_name}_{i}.jpg'),
                        frame
                    )

    def test_error_handling(self, setup_test_env, capture_test_images):
        """测试错误处理"""
        images, pose_data = capture_test_images
        binder = setup_test_env['binder']
        deformer = setup_test_env['deformer']
        
        # 1. 测试无效输入
        invalid_inputs = [
            (None, pose_data['neutral']),  # 无效图像
            (images['neutral'], None),      # 无效姿态
            (np.zeros((10, 10, 3)), pose_data['neutral']),  # 尺寸过小的图像
            (images['neutral'], PoseData([], None, 0.0, 0.0))  # 空姿态数据
        ]
        
        for img, pose in invalid_inputs:
            try:
                regions = binder.create_binding(img, pose)
                assert regions == [], "应该返回空列表"
            except Exception as e:
                logger.info(f"预期的错误处理: {str(e)}")
        
        # 2. 测试边界条件
        neutral_img = images['neutral']
        neutral_pose = pose_data['neutral']
        regions = binder.create_binding(neutral_img, neutral_pose)
        
        # 修改姿态数据以测试边界情况
        modified_poses = []
        # 超出图像边界的姿态
        out_of_bounds = neutral_pose.landmarks.copy()
        out_of_bounds[0].x = 2.0  # 超出归一化坐标范围
        modified_poses.append(PoseData(out_of_bounds, None, 0.0, 1.0))
        
        # 低置信度的姿态
        low_confidence = neutral_pose.landmarks.copy()
        for lm in low_confidence:
            lm.visibility = 0.1
        modified_poses.append(PoseData(low_confidence, None, 0.0, 0.1))
        
        for mod_pose in modified_poses:
            result = deformer.deform(
                neutral_img,
                neutral_pose,
                neutral_img.copy(),
                mod_pose,
                regions
            )
            # 应该返回原始图像而不是失败
            assert result is not None, "应该返回有效结果"
            assert np.array_equal(result, neutral_img), "应该返回原始图像"

    def _visualize_regions(self, image: np.ndarray, regions: List[DeformRegion]) -> np.ndarray:
        """可视化绑定区域"""
        vis_image = image.copy()
        
        # 为不同类型的区域使用不同的颜色
        colors = {
            'body': (0, 255, 0),   # 绿色
            'face': (0, 0, 255)    # 红色
        }
        
        for region in regions:
            color = colors.get(region.type, (255, 255, 255))
            
            # 绘制区域轮廓
            if region.mask is not None:
                # 创建彩色遮罩
                colored_mask = np.zeros_like(vis_image)
                colored_mask[region.mask > 0] = color
                
                # 半透明叠加
                alpha = 0.3
                mask_bool = region.mask > 0
                vis_image[mask_bool] = cv2.addWeighted(
                    vis_image[mask_bool],
                    1 - alpha,
                    colored_mask[mask_bool],
                    alpha,
                    0
                )
            
            # 绘制控制点和连接线
            points = []
            for bp in region.binding_points:
                point = (region.center + bp.local_coords).astype(np.int32)
                points.append(point)
                cv2.circle(vis_image, tuple(point), 3, color, -1)
            
            # 如果有多个点，绘制连接线
            if len(points) >= 2:
                points = np.array(points)
                cv2.polylines(vis_image, [points], True, color, 1)
            
            # 添加区域名称标签
            label_position = (
                int(region.center[0]),
                int(region.center[1] - 10)
            )
            cv2.putText(vis_image, region.name, label_position,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis_image

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, '-v'])

import cv2
import numpy as np
from pose.detector import PoseDetector
from pose.pose_binding import PoseBinding
from pose.pose_deformer import PoseDeformer
import os
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_image(frame):
    """预处理图像以提高检测质量"""
    # 调整大小到合适尺寸
    target_size = (640, 480)
    frame = cv2.resize(frame, target_size)
    
    # 增强对比度
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl,a,b))
    frame = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return frame

def visualize_landmarks(frame, pose, name, output_path):
    """可视化关键点和连接"""
    debug_frame = frame.copy()
    
    # 绘制关键点
    for i, lm in enumerate(pose.landmarks):
        x = int(lm.x * frame.shape[1])
        y = int(lm.y * frame.shape[0])
        # 根据可见度调整颜色
        color = (0, int(255 * lm.visibility), 0)
        # 绘制点
        cv2.circle(debug_frame, (x, y), 4, color, -1)
        # 添加索引标签
        cv2.putText(debug_frame, str(i), (x+5, y+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # 绘制连接线（使用POSE_CONFIG中的connections）
    connections = {
        'torso': [11, 12, 23, 24],  # 躯干
        'left_arm': [11, 13, 15],   # 左臂
        'right_arm': [12, 14, 16],  # 右臂
        'left_leg': [23, 25, 27],   # 左腿
        'right_leg': [24, 26, 28]   # 右腿
    }
    
    for part_name, indices in connections.items():
        for i in range(len(indices)-1):
            if (indices[i] < len(pose.landmarks) and 
                indices[i+1] < len(pose.landmarks)):
                pt1 = pose.landmarks[indices[i]]
                pt2 = pose.landmarks[indices[i+1]]
                if pt1.visibility > 0.5 and pt2.visibility > 0.5:
                    x1 = int(pt1.x * frame.shape[1])
                    y1 = int(pt1.y * frame.shape[0])
                    x2 = int(pt2.x * frame.shape[1])
                    y2 = int(pt2.y * frame.shape[0])
                    cv2.line(debug_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # 添加标题
    cv2.putText(debug_frame, f"{name} Frame Landmarks", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imwrite(output_path, debug_frame)

def test_pose_transform():
    # 1. 读取图像
    base_dir = os.path.dirname(os.path.dirname(__file__))
    ph1_path = os.path.join(base_dir, 'tests', 'photos', 'ph1.jpg')
    ph2_path = os.path.join(base_dir, 'tests', 'photos', 'ph2.jpg')
    
    initial_frame = cv2.imread(ph1_path)
    target_frame = cv2.imread(ph2_path)
    
    if initial_frame is None or target_frame is None:
        raise ValueError("无法读取图像文件")
        
    # 预处理图像
    initial_frame = preprocess_image(initial_frame)
    target_frame = preprocess_image(target_frame)
    
    # 保存预处理后的图像用于调试
    debug_dir = os.path.join(base_dir, 'tests', 'photos', 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, 'preprocessed_initial.jpg'), initial_frame)
    cv2.imwrite(os.path.join(debug_dir, 'preprocessed_target.jpg'), target_frame)

    # 2. 初始化组件
    detector = PoseDetector()
    binder = PoseBinding()
    deformer = PoseDeformer()
    
    try:
        # 3. 检测两张图片的姿态
        initial_pose = detector.detect(initial_frame)
        if initial_pose is None:
            raise ValueError("初始帧姿态检测失败")
        logger.info(f"初始帧置信度: {initial_pose.confidence}")
            
        target_pose = detector.detect(target_frame)
        if target_pose is None:
            raise ValueError("目标帧姿态检测失败")
        logger.info(f"目标帧置信度: {target_pose.confidence}")
        
        # 检查姿态数据的有效性
        if len(initial_pose.landmarks) < 33 or len(target_pose.landmarks) < 33:
            raise ValueError(f"关键点数量不足: 初始帧={len(initial_pose.landmarks)}, 目标帧={len(target_pose.landmarks)}")
            
        # 4. 创建变形区域
        try:
            regions = binder.create_binding(initial_frame, initial_pose)
            if not regions:
                logger.warning("没有创建任何变形区域")
                # 保存调试图像
                debug_frame = initial_frame.copy()
                for i, lm in enumerate(initial_pose.landmarks):
                    pt = (int(lm.x * initial_frame.shape[1]), int(lm.y * initial_frame.shape[0]))
                    cv2.circle(debug_frame, pt, 3, (0, 255, 0), -1)
                    cv2.putText(debug_frame, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.imwrite(os.path.join(debug_dir, 'debug_landmarks.jpg'), debug_frame)
            else:
                logger.info(f"成功创建变形区域: {len(regions)} 个区域")
                if regions:
                    logger.info("成功创建的区域:")
                    for region_name, region in regions.items():
                        logger.info(f"区域 {region_name}:")
                        logger.info(f"  中心点: {region.center}")
                        logger.info(f"  绑定点数量: {len(region.binding_points)}")
                        # 保存区域蒙版可视化
                        if region.mask is not None:
                            mask_vis = initial_frame.copy()
                            mask_vis[region.mask > 0] = [0, 255, 0]
                            cv2.imwrite(os.path.join(debug_dir, f'region_{region_name}_mask.jpg'), mask_vis)
        except Exception as e:
            logger.error(f"创建变形区域失败: {str(e)}")
            raise
        
        # 5. 应用变形
        try:
            result = deformer.deform_frame(initial_frame, regions, target_pose)
            logger.info("变形操作成功完成")
        except ValueError as e:
            logger.error(f"变形失败: {str(e)}")
            # 保存调试信息
            debug_info = {
                'initial_confidence': initial_pose.confidence,
                'target_confidence': target_pose.confidence,
                'initial_landmarks': len(initial_pose.landmarks),
                'target_landmarks': len(target_pose.landmarks),
                'visible_points': sum(1 for lm in target_pose.landmarks if lm.visibility > 0.5)
            }
            logger.error(f"调试信息: {debug_info}")
            raise
        
        # 6. 保存结果
        output_path = os.path.join(base_dir, 'tests', 'photos', 'result.jpg')
        cv2.imwrite(output_path, result)
        logger.info(f"结果已保存到: {output_path}")
        
        # 7. 可选：保存中间结果用于调试
        debug_dir = os.path.join(base_dir, 'tests', 'photos', 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        
        # 保存带关键点的图像
        for name, frame, pose in [('initial', initial_frame, initial_pose), 
                                ('target', target_frame, target_pose)]:
            output_path = os.path.join(debug_dir, f'{name}_landmarks.jpg')
            visualize_landmarks(frame, pose, name, output_path)
        
        # 保存变形结果和原始图像的对比
        comparison = np.hstack([initial_frame, result, target_frame])
        cv2.putText(comparison, "Initial", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(comparison, "Deformed", (initial_frame.shape[1] + 10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(comparison, "Target", (initial_frame.shape[1] * 2 + 10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(debug_dir, 'comparison.jpg'), comparison)
        
        # 输出更详细的调试信息
        logger.info("姿态检测详细信息:")
        for i, (lm_init, lm_target) in enumerate(zip(initial_pose.landmarks, target_pose.landmarks)):
            logger.info(f"关键点 {i}: 初始可见度={lm_init.visibility:.2f}, 目标可见度={lm_target.visibility:.2f}")
            
        # 检查区域创建
        if not regions:
            logger.warning("没有创建任何变形区域，检查区域创建条件")
            logger.info("初始姿态关键点位置:")
            for i, lm in enumerate(initial_pose.landmarks):
                logger.info(f"关键点 {i}: x={lm.x:.2f}, y={lm.y:.2f}, vis={lm.visibility:.2f}")
        
    finally:
        detector.release()

if __name__ == "__main__":
    test_pose_transform() 
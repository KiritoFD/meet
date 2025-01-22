import numpy as np
import torch
import time
import psutil
import cv2
from skimage.metrics import structural_similarity
from typing import List, Dict
from pose.deformer import PoseDeformer, SkeletonBinding

def create_test_binding():
    """创建测试用的骨骼绑定"""
    # 创建一个简单的测试图像
    reference_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 创建测试关键点
    landmarks = [
        {'x': 100, 'y': 100},
        {'x': 200, 'y': 200},
        {'x': 300, 'y': 300}
    ]
    
    # 创建测试骨骼结构
    bones = [
        {'start_idx': 0, 'end_idx': 1, 'children': [1]},
        {'start_idx': 1, 'end_idx': 2, 'children': []}
    ]
    
    # 创建测试网格点和权重
    mesh_points = np.array([[x, y] for x in range(0, 640, 32) for y in range(0, 480, 32)])
    weights = np.random.rand(len(mesh_points), len(bones))
    weights = weights / weights.sum(axis=1, keepdims=True)
    
    return SkeletonBinding(
        reference_frame=reference_frame,
        landmarks=landmarks,
        bones=bones,
        weights=weights,
        mesh_points=mesh_points,
        valid=True
    )

def compute_deformation_accuracy(result: np.ndarray, expected: np.ndarray) -> float:
    """计算变形精度"""
    if result.shape != expected.shape:
        return 0.0
    diff = np.abs(result.astype(float) - expected.astype(float))
    return 1.0 - (diff.mean() / 255.0)

def measure_edge_artifacts(image: np.ndarray) -> float:
    """测量边缘锯齿"""
    edges = cv2.Canny(image, 100, 200)
    return cv2.countNonZero(edges) / (image.shape[0] * image.shape[1])

def measure_texture_distortion(result: np.ndarray, reference: np.ndarray) -> float:
    """测量纹理失真"""
    if result.shape != reference.shape:
        return 1.0
    ssim = structural_similarity(result, reference, multichannel=True)
    return 1.0 - ssim

def compute_frame_jitter(current: np.ndarray, previous: np.ndarray) -> float:
    """计算帧间抖动"""
    if current.shape != previous.shape:
        return float('inf')
    diff = np.abs(current.astype(float) - previous.astype(float))
    return diff.mean()

def measure_deformation_continuity(image: np.ndarray) -> float:
    """测量变形连续性"""
    gradients = np.gradient(image.astype(float))
    gradient_magnitude = np.sqrt(gradients[0]**2 + gradients[1]**2)
    return 1.0 - (gradient_magnitude.mean() / 255.0)

def check_boundary_integrity(image: np.ndarray) -> float:
    """检查边界完整性"""
    border = np.concatenate([
        image[0, :],
        image[-1, :],
        image[:, 0],
        image[:, -1]
    ])
    return 1.0 - (np.count_nonzero(border == 0) / len(border))

# 创建测试数据
test_pose = [
    {'rotation': 45, 'translation': [10, 10]},
    {'rotation': -30, 'translation': [-5, 5]},
    {'rotation': 0, 'translation': [0, 0]}
]

test_pose_sequence = [
    [{'rotation': angle, 'translation': [x, x]} for angle, x in zip([0, 0, 0], [0, 0, 0])],
    [{'rotation': angle, 'translation': [x, x]} for angle, x in zip([10, -5, 0], [2, -1, 0])],
    [{'rotation': angle, 'translation': [x, x]} for angle, x in zip([20, -10, 0], [4, -2, 0])]
]

# 创建预期结果
expected_result = np.zeros((480, 640, 3), dtype=np.uint8)

def test_performance():
    """性能指标测试"""
    binding = create_test_binding()
    deformer = PoseDeformer(binding)
    
    # 测试单帧处理时间
    start_time = time.time()
    result = deformer.transform_frame(test_pose)
    process_time = time.time() - start_time
    assert process_time < 0.01, f"处理时间超过10ms: {process_time*1000:.2f}ms"
    
    # 测试GPU内存占用
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        assert memory_used < 500, f"GPU内存占用超过500MB: {memory_used:.2f}MB"
    
    # 测试CPU使用率
    cpu_percent = psutil.cpu_percent()
    assert cpu_percent < 20, f"CPU使用率超过20%: {cpu_percent}%"

def test_quality():
    """质量指标测试"""
    binding = create_test_binding()
    deformer = PoseDeformer(binding)
    
    result = deformer.transform_frame(test_pose)
    
    # 测试变形精度
    accuracy = compute_deformation_accuracy(result, expected_result)
    assert accuracy > 0.9, f"变形精度低于90%: {accuracy*100:.2f}%"
    
    # 测试边缘锯齿
    edge_error = measure_edge_artifacts(result)
    assert edge_error < 1.0, f"边缘锯齿超过1px: {edge_error:.2f}px"
    
    # 测试纹理失真
    texture_error = measure_texture_distortion(result, binding.reference_frame)
    assert texture_error < 0.05, f"纹理失真超过5%: {texture_error*100:.2f}%"

def test_stability():
    """稳定性指标测试"""
    binding = create_test_binding()
    deformer = PoseDeformer(binding)
    
    # 测试帧间抖动
    prev_result = None
    max_jitter = 0
    
    for pose in test_pose_sequence:
        result = deformer.transform_frame(pose)
        if prev_result is not None:
            jitter = compute_frame_jitter(result, prev_result)
            max_jitter = max(max_jitter, jitter)
        prev_result = result.copy()
    
    assert max_jitter < 0.5, f"帧间抖动超过0.5px: {max_jitter:.2f}px"
    
    # 测试变形连续性
    continuity = measure_deformation_continuity(result)
    assert continuity > 0.95, f"变形连续性低于95%: {continuity*100:.2f}%"
    
    # 测试边界完整性
    boundary_integrity = check_boundary_integrity(result)
    assert boundary_integrity == 1.0, f"边界不完整: {boundary_integrity*100:.2f}%" 
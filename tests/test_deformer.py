import unittest
import numpy as np
import torch
import time
from pose.deformer import PoseDeformer, DeformationConfig
from pose.binding import SkeletonBinding
import cv2

class TestPoseDeformer(unittest.TestCase):
    def setUp(self):
        """测试初始化"""
        # 创建模拟的骨骼绑定数据
        self.reference_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.landmarks = [{'x': 0.5, 'y': 0.5} for _ in range(10)]
        self.binding = self._create_mock_binding()
        self.deformer = PoseDeformer(self.binding)
        
    def _create_mock_binding(self) -> SkeletonBinding:
        """创建模拟的骨骼绑定"""
        binding = SkeletonBinding(
            reference_frame=self.reference_frame,
            landmarks=self.landmarks,
            bones=[],  # 简化的骨骼结构
            weights=np.random.rand(100, 10),  # 100个网格点，10个骨骼
            mesh_points=np.random.rand(100, 2) * [640, 480],
            valid=True
        )
        return binding

    def test_performance_requirements(self):
        """测试性能要求"""
        print("\n性能测试:")
        
        # 准备测试数据
        current_pose = [{'x': 0.5 + np.random.rand()*0.1, 
                        'y': 0.5 + np.random.rand()*0.1} 
                       for _ in range(10)]
        
        # 预热
        for _ in range(10):
            self.deformer.transform_frame(current_pose)
        
        # 测试处理时间
        times = []
        for _ in range(100):
            start = time.time()
            self.deformer.transform_frame(current_pose)
            times.append((time.time() - start) * 1000)  # 转换为毫秒
        
        avg_time = np.mean(times)
        print(f"平均处理时间: {avg_time:.2f}ms")
        self.assertLess(avg_time, 10.0, "单帧处理时间应小于10ms")
        
        # 测试GPU内存使用
        if torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            print(f"GPU内存使用: {memory_usage:.2f}MB")
            self.assertLess(memory_usage, 500, "GPU内存使用应小于500MB")
        
        # 测试CPU使用率
        report = self.deformer.get_performance_report()
        print(f"性能报告: {report}")

    def test_quality_requirements(self):
        """测试质量要求"""
        print("\n质量测试:")
        
        # 准备测试数据
        current_pose = [{'x': 0.5, 'y': 0.5} for _ in range(10)]
        
        # 测试变形精度
        result = self.deformer.transform_frame(current_pose)
        self.assertEqual(result.shape, self.reference_frame.shape)
        
        # 测试边缘质量
        edges = cv2.Canny(result, 100, 200)
        jaggy_count = np.sum(edges > 0)
        print(f"边缘像素数: {jaggy_count}")
        
        # 测试纹理质量
        if len(self.deformer.frame_buffer) > 0:
            prev_frame = self.deformer.frame_buffer[-1]
            texture_diff = np.mean(np.abs(result - prev_frame))
            print(f"纹理差异: {texture_diff:.2f}")
            self.assertLess(texture_diff, 0.05 * 255, "纹理失真应小于5%")

    def test_stability_requirements(self):
        """测试稳定性要求"""
        print("\n稳定性测试:")
        
        # 测试异常输入
        invalid_pose = [{'x': 0.5} for _ in range(10)]  # 缺少y坐标
        try:
            result = self.deformer.transform_frame(invalid_pose)
            print("成功处理异常输入")
        except Exception as e:
            self.fail(f"异常输入处理失败: {e}")
        
        # 测试连续帧稳定性
        results = []
        for _ in range(10):
            current_pose = [{'x': 0.5 + np.random.rand()*0.01, 
                           'y': 0.5 + np.random.rand()*0.01} 
                          for _ in range(10)]
            results.append(self.deformer.transform_frame(current_pose))
        
        # 计算帧间差异
        diffs = []
        for i in range(1, len(results)):
            diff = np.mean(np.abs(results[i] - results[i-1]))
            diffs.append(diff)
        
        avg_diff = np.mean(diffs)
        print(f"平均帧间差异: {avg_diff:.2f}")
        self.assertLess(avg_diff, 0.5, "帧间抖动应小于0.5px")

    def test_cache_system(self):
        """测试缓存系统"""
        print("\n缓存系统测试:")
        
        # 禁用缓存的性能
        self.deformer.config.enable_cache = False
        start = time.time()
        for _ in range(10):
            current_pose = [{'x': 0.5, 'y': 0.5} for _ in range(10)]
            self.deformer.transform_frame(current_pose)
        no_cache_time = time.time() - start
        
        # 启用缓存的性能
        self.deformer.config.enable_cache = True
        start = time.time()
        for _ in range(10):
            current_pose = [{'x': 0.5, 'y': 0.5} for _ in range(10)]
            self.deformer.transform_frame(current_pose)
        cache_time = time.time() - start
        
        print(f"无缓存时间: {no_cache_time:.3f}s")
        print(f"有缓存时间: {cache_time:.3f}s")
        self.assertLess(cache_time, no_cache_time, "缓存应该提高性能")

    def test_adaptive_batch(self):
        """测试自适应批处理"""
        print("\n自适应批处理测试:")
        
        self.deformer.config.enable_adaptive_batch = True
        batch_sizes = []
        
        for _ in range(20):
            current_pose = [{'x': 0.5 + np.random.rand()*0.1, 
                           'y': 0.5 + np.random.rand()*0.1} 
                          for _ in range(10)]
            self.deformer.transform_frame(current_pose)
            batch_sizes.append(self.deformer._get_adaptive_batch_size())
        
        print(f"批处理大小变化: {batch_sizes}")
        self.assertTrue(len(set(batch_sizes)) > 1, "批处理大小应该自适应调整")

def main():
    unittest.main(verbosity=2)

if __name__ == '__main__':
    main() 
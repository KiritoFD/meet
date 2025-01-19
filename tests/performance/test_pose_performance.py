import pytest
import time
import psutil
import os
from pose.pose_deformer import PoseDeformer
from pose.pose_binding import PoseBinding
from connect.pose_sender import PoseSender

class TestPosePerformance:
    def test_memory_leak(self):
        """测试内存泄漏"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        deformer = PoseDeformer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 执行大量变形操作
        for _ in range(1000):
            pose = self._create_test_pose()
            deformer.deform_frame(frame, pose)
            
        # 强制GC
        import gc
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_growth = (final_memory - initial_memory) / (1024 * 1024)  # MB
        assert memory_growth < 50  # 内存增长不超过50MB

    def test_cpu_usage(self):
        """测试CPU使用"""
        process = psutil.Process(os.getpid())
        deformer = PoseDeformer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        cpu_percentages = []
        for _ in range(100):
            start_cpu = process.cpu_percent()
            pose = self._create_test_pose()
            deformer.deform_frame(frame, pose)
            end_cpu = process.cpu_percent()
            cpu_percentages.append(end_cpu - start_cpu)
        
        avg_cpu = sum(cpu_percentages) / len(cpu_percentages)
        assert avg_cpu < 50  # 平均CPU使用率不超过50% 
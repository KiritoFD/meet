import time
import numpy as np
from contextlib import contextmanager
import cv2

@contextmanager
def measure_time():
    """测量代码块执行时间的上下文管理器"""
    start = time.time()
    yield
    end = time.time()
    return end - start

def generate_test_sequence(frame_count=30):
    """生成测试序列"""
    frames = []
    poses = []
    for i in range(frame_count):
        # 创建测试帧
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        angle = i * (360 / frame_count)
        x = int(320 + 100 * np.cos(np.radians(angle)))
        y = int(240 + 100 * np.sin(np.radians(angle)))
        cv2.circle(frame, (x, y), 20, (255, 255, 255), -1)
        frames.append(frame)
        
        # 创建对应的姿态数据
        pose = {
            'landmarks': [{
                'x': x / 640,
                'y': y / 480,
                'z': 0.0,
                'visibility': 1.0
            } for _ in range(33)]
        }
        poses.append(pose)
    
    return frames, poses 
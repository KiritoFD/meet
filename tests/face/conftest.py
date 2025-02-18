import pytest
import os
import cv2
import numpy as np

@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """设置测试环境"""
    # 创建测试图片目录
    test_images_dir = os.path.join(os.path.dirname(__file__), 'test_images')
    os.makedirs(test_images_dir, exist_ok=True)
    
    # 清理旧的测试图片
    for file in os.listdir(test_images_dir):
        if file.endswith('.jpg'):
            os.remove(os.path.join(test_images_dir, file)) 
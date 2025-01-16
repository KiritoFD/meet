import cv2
import numpy as np
import torch
from pathlib import Path
import sys
import logging
from tqdm import tqdm

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from pose.smoother import FrameSmoother

def process_image(image_path, output_path=None):
    """处理单张图片"""
    # 初始化平滑器
    smoother = FrameSmoother(
        model_path='models/NAFNet-GoPro-width32.pth',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        downsample_factor=1.0  # 使用原始分辨率
    )
    
    # 读取图片
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"无法读取图片: {image_path}")
        return
        
    # 添加一些噪声进行测试
    noisy = image.astype(np.float32) + np.random.normal(0, 25, image.shape)
    noisy = noisy.clip(0, 255).astype(np.uint8)
    
    # 处理图片
    smoothed = smoother.smooth_frame(noisy)
    
    # 拼接对比图
    comparison = np.hstack([noisy, smoothed])
    
    # 显示结果
    cv2.namedWindow('Comparison', cv2.WINDOW_NORMAL)
    cv2.imshow('Comparison', comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存结果
    if output_path:
        cv2.imwrite(str(output_path), comparison)
        print(f"结果已保存到: {output_path}")

if __name__ == "__main__":
    # 可以替换为你的图片路径
    image_path = "test_images/test.jpg"  # 替换为你的图片路径
    output_path = "results/comparison.jpg"
    
    # 创建输出目录
    Path(output_path).parent.mkdir(exist_ok=True)
    
    # 处理图片
    process_image(image_path, output_path) 
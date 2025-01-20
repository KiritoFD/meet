import torch
import os
from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from pose.smoother import NAFNet

def create_lightweight_model():
    """创建轻量级NAFNet模型"""
    print("创建轻量级NAFNet模型...")
    
    # 创建模型
    model = NAFNet(
        img_channel=3,
        width=8,  # 极小的通道数
        middle_blk_num=1,  # 单个中间块
        enc_blk_nums=[1],  # 单层编码器
        dec_blk_nums=[1]   # 单层解码器
    )
    
    # 创建保存目录
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / 'nafnet_smoother.pth'
    
    # 保存模型
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")
    
    # 验证文件大小
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"模型大小: {size_mb:.1f}MB")

if __name__ == "__main__":
    create_lightweight_model() 
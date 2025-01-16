import cv2
import torch
import numpy as np
from pathlib import Path
import sys
import logging
from tqdm import tqdm

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from pose.smoother import NAFNet

def test_model(model_path='models/NAFNet-GoPro-width32.pth'):
    """测试模型效果"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # 加载模型 - 使用GoPro配置
    model = NAFNet(
        img_channel=3,
        width=32,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2]
    ).to(device)
    
    # 加载预训练权重
    state_dict = torch.load(model_path, map_location=device)
    if 'params' in state_dict:
        state_dict = state_dict['params']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    if device.type == 'cuda':
        model = model.half()  # 使用FP16加速
    
    if device.type == 'cuda':
        # 使用 TensorRT 加速（如果可用）
        try:
            from torch2trt import torch2trt
            x = torch.randn(1, 3, 480, 640).to(device)
            model_trt = torch2trt(model, [x])
            model = model_trt
            print("使用 TensorRT 加速")
        except:
            print("TensorRT 不可用，使用普通 CUDA 模式")
    
    # 创建测试窗口
    cv2.namedWindow('Test', cv2.WINDOW_NORMAL)
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    # 预分配内存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        stream = torch.cuda.Stream()
    
    try:
        with torch.cuda.amp.autocast():  # 使用混合精度推理
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 添加噪声进行测试
                noisy = frame.astype(np.float32) + np.random.normal(0, 25, frame.shape)
                noisy = noisy.clip(0, 255).astype(np.uint8)
                
                # 处理图像
                with torch.no_grad():
                    if device.type == 'cuda':
                        with torch.cuda.stream(stream):
                            tensor = torch.from_numpy(noisy).float().to(device, non_blocking=True) / 255.0
                            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
                            output = model(tensor)
                            stream.synchronize()
                    else:
                        tensor = torch.from_numpy(noisy).float().to(device) / 255.0
                        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
                        output = model(tensor)
                    
                    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    output = (output * 255).clip(0, 255).astype(np.uint8)
                
                # 显示结果
                display = np.hstack([noisy, output])
                cv2.imshow('Test', display)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

if __name__ == "__main__":
    test_model() 
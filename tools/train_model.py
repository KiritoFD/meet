import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import numpy as np
from pathlib import Path
import sys
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import json
# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from pose.smoother import NAFNet

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NoiseDataset(Dataset):
    """噪声数据集"""
    def __init__(self, size=(480, 640), num_samples=2000):
        self.size = size
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
        
    def add_noise(self, image):
        """添加多种噪声"""
        # 高斯噪声
        noise = np.random.normal(0, 25, image.shape).astype(np.float32)
        noisy = image + noise
        
        # 运动模糊
        kernel_size = np.random.choice([3, 5, 7])
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size//2, :] = 1.0 / kernel_size
        noisy = cv2.filter2D(noisy, -1, kernel)
        
        # 压缩伪影
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), np.random.randint(60, 95)]
        _, encoded = cv2.imencode('.jpg', noisy, encode_param)
        noisy = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        
        # 随机亮度和对比度变化
        alpha = np.random.uniform(0.8, 1.2)
        beta = np.random.uniform(-30, 30)
        noisy = cv2.convertScaleAbs(noisy, alpha=alpha, beta=beta)
        
        return noisy.clip(0, 255).astype(np.uint8)
        
    def __getitem__(self, idx):
        # 生成干净图像（使用渐变或简单图案）
        clean = np.zeros((self.size[0], self.size[1], 3), dtype=np.uint8)
        cv2.rectangle(clean, (0, 0), (self.size[1], self.size[0]), 
                     (np.random.randint(0, 255),)*3, -1)
        cv2.circle(clean, (self.size[1]//2, self.size[0]//2), 
                  np.random.randint(50, 200), 
                  (np.random.randint(0, 255),)*3, -1)
        
        # 添加噪声
        noisy = self.add_noise(clean.copy())
        
        # 转换为张量
        clean = torch.from_numpy(clean).float() / 255.0
        noisy = torch.from_numpy(noisy).float() / 255.0
        
        # 调整通道顺序
        clean = clean.permute(2, 0, 1)
        noisy = noisy.permute(2, 0, 1)
        
        return noisy, clean

def save_validation_images(model, val_loader, device, epoch, save_dir):
    """保存验证图像"""
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    with torch.no_grad():
        for i, (noisy, clean) in enumerate(val_loader):
            if i >= 5:  # 只保存前5个样本
                break
                
            noisy = noisy.to(device)
            output = model(noisy)
            
            # 转换回numpy
            noisy = noisy[0].permute(1, 2, 0).cpu().numpy() * 255
            clean = clean[0].permute(1, 2, 0).cpu().numpy() * 255
            output = output[0].permute(1, 2, 0).cpu().numpy() * 255
            
            # 拼接图像
            comparison = np.hstack([noisy, output, clean])
            comparison = comparison.astype(np.uint8)
            
            # 保存图像
            cv2.imwrite(str(save_dir / f'epoch_{epoch}_sample_{i}.jpg'), 
                       cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

def train_model(epochs=100, batch_size=16, learning_rate=0.001):
    """训练模型"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(f'models/nafnet_{timestamp}')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建模型
    model = NAFNet(
        img_channel=3,
        width=8,
        middle_blk_num=1,
        enc_blk_nums=[1],
        dec_blk_nums=[1]
    ).to(device)
    
    # 创建数据集
    dataset = NoiseDataset()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, 
                          shuffle=False, num_workers=4)
    
    # 定义损失函数和优化器
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
                                                   patience=5, factor=0.5)
    
    # 记录训练历史
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_loss = float('inf')
    
    # 训练循环
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for noisy, clean in pbar:
                noisy = noisy.to(device)
                clean = clean.to(device)
                
                optimizer.zero_grad()
                output = model(noisy)
                loss = criterion(output, clean)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy = noisy.to(device)
                clean = clean.to(device)
                output = model(noisy)
                val_loss += criterion(output, clean).item()
        
        val_loss /= len(val_loader)
        
        # 更新学习率
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        
        logger.info(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, "
                   f"Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
        
        # 保存验证图像
        if (epoch + 1) % 10 == 0:
            save_validation_images(model, val_loader, device, epoch+1, 
                                save_dir / 'validation_images')
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_dir / 'nafnet_smoother.pth')
            
            # 同时复制一份到models目录
            torch.save(model.state_dict(), Path('models/nafnet_smoother.pth'))
            logger.info(f"保存最佳模型，验证损失: {best_loss:.4f}")
    
    # 保存训练历史
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f)
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['lr'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png')
    plt.close()

if __name__ == "__main__":
    train_model() 
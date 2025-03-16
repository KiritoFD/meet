import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch is not available. Using NumPy fallback for frame smoothing.")

import torch.nn as nn
import cv2
import logging
from typing import Optional, List
from collections import deque
import os
from contextlib import nullcontext
from .types import PoseData, Landmark

logger = logging.getLogger(__name__)

class NAFNet(nn.Module):
    """超轻量级NAFNet，专注于CPU实时处理"""
    def __init__(self, img_channel=3, width=8, middle_blk_num=1,
                 enc_blk_nums=[1], dec_blk_nums=[1]):  # 极简化结构
        super().__init__()
        
        # 使用分组卷积减少计算量
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1, groups=1)
        
        # 极简编码器
        self.encoders = nn.ModuleList()
        chan = width
        for _ in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    # 使用步长卷积替代池化
                    nn.Conv2d(chan, chan*2, 3, 2, 1, groups=chan),
                    nn.ReLU(inplace=True),  # 使用ReLU替代GELU
                    # 使用1x1卷积进行特征融合
                    nn.Conv2d(chan*2, chan*2, 1)
                )
            )
            chan = chan * 2
            
        # 轻量级中间块
        self.middle_blks = nn.ModuleList([
            nn.Sequential(
                # 深度可分离卷积
                nn.Conv2d(chan, chan, 3, 1, 1, groups=chan),
                nn.ReLU(inplace=True),
                nn.Conv2d(chan, chan, 1)
            ) for _ in range(middle_blk_num)
        ])
            
        # 极简解码器
        self.decoders = nn.ModuleList()
        for _ in dec_blk_nums:
            self.decoders.append(
                nn.Sequential(
                    # 使用插值上采样替代反卷积
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(chan, chan//2, 3, 1, 1, groups=chan//2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(chan//2, chan//2, 1)
                )
            )
            chan = chan // 2
            
        self.ending = nn.Conv2d(width, img_channel, 1)

    def forward(self, x):
        x = self.intro(x)
        encs = []
        
        # 编码
        for encoder in self.encoders:
            x = encoder(x)
            encs.append(x)
            
        # 中间处理
        for middle_blk in self.middle_blks:
            x = middle_blk(x)
            
        # 解码
        for decoder in self.decoders:
            if encs:
                x = x + encs.pop()
            x = decoder(x)
            
        x = self.ending(x)
        return x

class FrameSmoother:
    """
    Smooths frames or keypoints over time to reduce jitter.
    Falls back to NumPy implementation if PyTorch is not available.
    """
    def __init__(self, window_size=5, alpha=0.7):
        self.window_size = window_size
        self.alpha = alpha
        self.history = []
        self.torch_available = TORCH_AVAILABLE
        logger.info(f"Frame smoother initialized. Using PyTorch: {self.torch_available}")

    def smooth(self, data):
        """
        Apply exponential moving average smoothing to the data.
        
        Args:
            data: Array-like data to smooth (can be numpy array or torch tensor)
            
        Returns:
            Smoothed data in the same format as input
        """
        if data is None:
            return None
        
        # Convert input to appropriate format based on available library
        if self.torch_available:
            # PyTorch implementation
            if not isinstance(data, torch.Tensor):
                data_tensor = torch.tensor(data, dtype=torch.float32)
            else:
                data_tensor = data.clone()
                
            # Apply smoothing
            if not self.history:
                self.history.append(data_tensor)
                return data
                
            smoothed = self.alpha * data_tensor + (1 - self.alpha) * self.history[-1]
            
            # Update history
            self.history.append(smoothed)
            if len(self.history) > self.window_size:
                self.history.pop(0)
                
            return smoothed.numpy() if not isinstance(data, torch.Tensor) else smoothed
            
        else:
            # NumPy implementation
            data_np = np.array(data, dtype=np.float32)
            
            # Apply smoothing
            if not self.history:
                self.history.append(data_np)
                return data
                
            smoothed = self.alpha * data_np + (1 - self.alpha) * self.history[-1]
            
            # Update history
            self.history.append(smoothed)
            if len(self.history) > self.window_size:
                self.history.pop(0)
                
            return smoothed

    def reset(self):
        """Reset the smoothing history"""
        self.history = []

    def _preprocess_frame(self, frame: np.ndarray):
        if TORCH_AVAILABLE:
            return torch.tensor(frame, dtype=torch.float32)
        else:
            return np.array(frame, dtype=np.float32)

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """优化的预处理"""
        # 保存原始尺寸
        self.orig_height, self.orig_width = frame.shape[:2]
        
        # 下采样
        if self.downsample_factor < 1.0:
            new_h = int(self.orig_height * self.downsample_factor)
            new_w = int(self.orig_width * self.downsample_factor)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
        # 转换为张量
        frame = frame.astype(np.float32) * (1.0 / 255.0)
        tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
        
        # 转移到设备
        if self.device == 'cuda':
            tensor = tensor.half()  # 使用FP16
        return tensor.to(self.device, non_blocking=True)
        
    def _postprocess_frame(self, tensor: torch.Tensor) -> np.ndarray:
        """优化的后处理"""
        with torch.no_grad():
            # 转回CPU并转换为numpy
            if tensor.dtype == torch.float16:
                tensor = tensor.float()
            tensor = tensor.clamp_(0, 1)
            frame = (tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            # 恢复原始尺寸
            if frame.shape[:2] != (self.orig_height, self.orig_width):
                frame = cv2.resize(frame, (self.orig_width, self.orig_height), 
                                 interpolation=cv2.INTER_LINEAR)
            return frame
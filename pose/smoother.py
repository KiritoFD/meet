import torch
import torch.nn as nn
import numpy as np
import cv2
import logging
from typing import Optional
from collections import deque
import os
from contextlib import nullcontext

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
    def __init__(self, model_path: str = 'models/NAFNet-GoPro-width32.pth',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 buffer_size: int = 3,
                 temporal_weight: float = 0.8,
                 downsample_factor: float = 0.5):
        self.device = device
        self.buffer_size = buffer_size
        self.temporal_weight = temporal_weight
        self.downsample_factor = downsample_factor
        self.frame_buffer = deque(maxlen=buffer_size)
        self.orig_width = None
        self.orig_height = None
        
        # 初始化模型
        try:
            self.model = NAFNet(
                img_channel=3,
                width=32,
                middle_blk_num=12,
                enc_blk_nums=[2, 2, 4, 8],
                dec_blk_nums=[2, 2, 2, 2]
            ).to(device)
            
            if os.path.exists(model_path):
                try:
                    state_dict = torch.load(model_path, map_location=device)
                    if 'params' in state_dict:
                        state_dict = state_dict['params']
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                    self.model.load_state_dict(state_dict, strict=False)
                    logger.info(f"NAFNet模型加载成功，使用设备: {device}")
                except Exception as e:
                    logger.warning(f"加载预训练权重失败: {e}")
            
            self.model.eval()
            
            # CUDA优化
            if device == 'cuda':
                self.model = self.model.half()  # 使用FP16
                torch.backends.cudnn.benchmark = True
                
        except Exception as e:
            logger.error(f"NAFNet模型初始化失败: {e}")
            self.model = None
            
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
        
    def _apply_temporal_smooth(self, current_frame: np.ndarray) -> np.ndarray:
        """时间域平滑"""
        if not self.frame_buffer:
            self.frame_buffer.append(current_frame.copy())
            return current_frame
            
        # 使用指数加权移动平均
        smoothed = cv2.addWeighted(
            current_frame, self.temporal_weight,
            self.frame_buffer[-1], 1 - self.temporal_weight,
            0
        )
        
        # 存储当前帧的副本
        self.frame_buffer.append(smoothed.copy())
        return smoothed
        
    @torch.no_grad()
    def smooth_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """优化的帧处理"""
        if frame is None or self.model is None:
            return frame
            
        try:
            # 空间域平滑
            with torch.cuda.amp.autocast() if self.device == 'cuda' else nullcontext():
                tensor = self._preprocess_frame(frame)
                output = self.model(tensor)
                smoothed = self._postprocess_frame(output)
            
            # 时间域平滑
            smoothed = self._apply_temporal_smooth(smoothed)
            
            return smoothed
            
        except Exception as e:
            logger.error(f"帧平滑失败: {e}")
            return frame
            
    def reset(self):
        """重置平滑器状态"""
        self.frame_buffer.clear()
        torch.cuda.empty_cache() if self.device == 'cuda' else None 
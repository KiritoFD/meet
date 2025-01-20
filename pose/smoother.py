import torch
import torch.nn as nn
import numpy as np
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
    """基础帧平滑器"""
    
    def __init__(self, 
                 temporal_weight: float = 0.8,
                 spatial_weight: float = 0.5,
                 **kwargs):  # 添加 **kwargs 支持扩展参数
        """初始化平滑器
        
        Args:
            temporal_weight: 时间平滑权重 (0-1)
            spatial_weight: 空间平滑权重 (0-1)
            **kwargs: 扩展参数
        """
        self.temporal_weight = temporal_weight
        self.spatial_weight = spatial_weight
        self.frame_buffer = []
        
    def _smooth_points(self, points: np.ndarray) -> np.ndarray:
        """平滑关键点坐标
        
        Args:
            points: 关键点坐标数组 [N, 3]
            
        Returns:
            平滑后的坐标数组 [N, 3]
        """
        if not self.frame_buffer:
            self.frame_buffer.append(points.copy())
            return points
            
        # 时间域平滑
        smoothed = (
            self.temporal_weight * points + 
            (1 - self.temporal_weight) * self.frame_buffer[-1]
        )
        
        # 空间域平滑（可选）
        if self.spatial_weight > 0:
            kernel = np.array([0.25, 0.5, 0.25])  # 简单的1D平滑核
            for i in range(3):  # x, y, z
                smoothed[:, i] = np.convolve(
                    smoothed[:, i],
                    kernel,
                    mode='same'
                )
        
        # 更新缓冲区
        self.frame_buffer.append(smoothed.copy())
        if len(self.frame_buffer) > 5:  # 保持固定大小的缓冲区
            self.frame_buffer.pop(0)
            
        return smoothed
        
    def smooth(self, pose_data: PoseData) -> PoseData:
        """平滑姿态数据
        
        Args:
            pose_data: 输入的姿态数据
            
        Returns:
            平滑后的姿态数据
        """
        # 获取关键点坐标数组
        points = pose_data.values
        
        # 应用平滑
        smoothed = self._smooth_points(points)
        
        # 创建新的姿态数据
        return PoseData(
            landmarks=[
                Landmark(x=x, y=y, z=z)
                for x, y, z in smoothed
            ],
            timestamp=pose_data.timestamp,
            confidence=pose_data.confidence
        )
        
    def reset(self):
        """重置平滑器状态"""
        self.frame_buffer.clear()

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
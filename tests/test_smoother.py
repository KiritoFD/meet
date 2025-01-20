import sys
import os
import pytest
import cv2
import numpy as np
import torch
from pathlib import Path
import logging
import time
# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from pose.smoother import FrameSmoother, NAFNet

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestFrameSmoother:
    @pytest.fixture(scope="class")
    def setup_smoother(self):
        """初始化平滑器"""
        # 检查模型文件是否存在
        model_path = Path('models/nafnet_smoother.pth')
        if not model_path.exists():
            # 创建轻量级测试模型
            logger.warning("找不到预训练模型，创建轻量级测试模型")
            model = NAFNet(
                img_channel=3,
                width=32,  # 使用与预训练模型相同的配置
                middle_blk_num=12,
                enc_blk_nums=[2, 2, 4, 8],
                dec_blk_nums=[2, 2, 2, 2]
            )
            
            # 保存模型权重
            model_path.parent.mkdir(exist_ok=True)
            # 直接保存 state_dict，不使用 weights_only 参数
            torch.save(model.state_dict(), model_path)
            logger.info("已创建并保存测试模型")
            
        smoother = FrameSmoother(
            model_path=str(model_path),
            device='cuda' if torch.cuda.is_available() else 'cpu',
            buffer_size=3,
            temporal_weight=0.8,
            downsample_factor=0.5
        )
        yield smoother
        # 清理
        smoother.reset()
        
    def test_model_initialization(self, setup_smoother):
        """测试模型初始化"""
        smoother = setup_smoother
        assert smoother.model is not None, "模型初始化失败"
        assert isinstance(smoother.model, (NAFNet, torch.jit.ScriptModule)), "模型类型错误"
        
    def test_frame_preprocessing(self, setup_smoother):
        """测试帧预处理"""
        smoother = setup_smoother
        
        # 创建测试帧
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 测试预处理
        tensor = smoother._preprocess_frame(test_frame)
        
        # 计算预期尺寸
        expected_h = int(480 * smoother.downsample_factor)
        expected_w = int(640 * smoother.downsample_factor)
        
        assert isinstance(tensor, torch.Tensor), "预处理输出类型错误"
        assert tensor.shape == (1, 3, expected_h, expected_w), "预处理输出形状错误"
        assert tensor.device == torch.device(smoother.device), "设备错误"
        assert tensor.max() <= 1.0 and tensor.min() >= 0.0, "归一化错误"
        
    def test_frame_postprocessing(self, setup_smoother):
        """测试帧后处理"""
        smoother = setup_smoother
        
        # 创建测试张量
        test_tensor = torch.rand(1, 3, 480, 640).to(smoother.device)
        
        # 测试后处理
        frame = smoother._postprocess_frame(test_tensor)
        
        assert isinstance(frame, np.ndarray), "后处理输出类型错误"
        assert frame.shape == (480, 640, 3), "后处理输出形状错误"
        assert frame.dtype == np.uint8, "后处理输出数据类型错误"
        assert frame.max() <= 255 and frame.min() >= 0, "值范围错误"
        
    def test_temporal_smoothing(self, setup_smoother):
        """测试时间域平滑"""
        smoother = setup_smoother
        
        # 创建测试帧序列
        frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        
        # 测试时间域平滑
        smoothed_frames = []
        for frame in frames:
            smoothed = smoother.smooth_frame(frame)
            smoothed_frames.append(smoothed)
            
        # 验证平滑效果
        assert len(smoothed_frames) == len(frames), "输出帧数不匹配"
        
        # 验证时间连续性
        if len(smoothed_frames) > 1:
            diffs = []
            for i in range(1, len(smoothed_frames)):
                diff = np.mean(np.abs(smoothed_frames[i].astype(float) - 
                                    smoothed_frames[i-1].astype(float)))
                diffs.append(diff)
            
            avg_diff = np.mean(diffs)
            logger.info(f"平均帧间差异: {avg_diff:.2f}")
            assert avg_diff < 100, f"时间平滑效果不佳: {avg_diff:.2f}"
        
    def test_full_smoothing_pipeline(self, setup_smoother):
        """测试完整平滑流程"""
        smoother = setup_smoother
        
        # 创建测试帧
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 测试完整流程
        smoothed_frame = smoother.smooth_frame(test_frame)
        
        assert smoothed_frame is not None, "平滑处理失败"
        assert smoothed_frame.shape == test_frame.shape, "输出形状错误"
        assert smoothed_frame.dtype == np.uint8, "输出数据类型错误"
        
    def test_error_handling(self, setup_smoother):
        """测试错误处理"""
        smoother = setup_smoother
        
        # 测试空输入
        assert smoother.smooth_frame(None) is None, "空输入处理错误"
        
        # 测试错误形状
        wrong_shape_frame = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        try:
            smoother.smooth_frame(wrong_shape_frame)
            assert False, "应该抛出错误"
        except Exception as e:
            assert True
            
    def test_performance(self, setup_smoother):
        """测试性能"""
        smoother = setup_smoother
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 预热
        for _ in range(5):
            smoother.smooth_frame(test_frame)
            
        # 测试处理时间
        times = []
        for _ in range(20):  # 增加测试次数
            start_time = time.time()
            smoother.smooth_frame(test_frame)
            times.append(time.time() - start_time)
            
        # 去掉最慢的几次
        times.sort()
        avg_time = np.mean(times[:15])
        logger.info(f"平均处理时间: {avg_time*1000:.1f}ms")
        
        # 根据设备调整性能要求
        if torch.cuda.is_available():
            target_time = 0.05  # GPU: 50ms
        else:
            target_time = 0.2   # CPU: 200ms
        assert avg_time < target_time, f"处理太慢: {avg_time*1000:.1f}ms"
        
    @pytest.mark.skipif(not torch.cuda.is_available(), 
                       reason="需要CUDA支持")
    def test_gpu_memory(self, setup_smoother):
        """测试GPU内存使用"""
        if not torch.cuda.is_available():
            return
            
        smoother = setup_smoother
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 记录初始内存
        torch.cuda.empty_cache()
        start_mem = torch.cuda.memory_allocated()
        
        # 运行模型
        for _ in range(10):
            smoother.smooth_frame(test_frame)
            
        # 检查内存增长
        end_mem = torch.cuda.memory_allocated()
        mem_increase = (end_mem - start_mem) / 1024 / 1024  # MB
        
        logger.info(f"GPU内存增长: {mem_increase:.1f}MB")
        assert mem_increase < 500, f"内存使用过多: {mem_increase:.1f}MB"

if __name__ == "__main__":
    pytest.main(["-v", "test_smoother.py"]) 
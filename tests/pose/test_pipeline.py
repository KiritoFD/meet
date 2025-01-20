import pytest
import numpy as np
import cv2
from pose import PosePipeline, PoseData

class TestPosePipeline:
    @pytest.fixture
    def pipeline(self):
        """创建测试用的处理管线"""
        config = {
            'smoother': {
                # 基础平滑参数
                'temporal_weight': 0.8,
                'spatial_weight': 0.5,
                
                # 变形平滑参数
                'deform_threshold': 30,
                'edge_width': 3,
                'motion_scale': 0.5,
                
                # 质量评估参数
                'quality_weights': {
                    'temporal': 0.4,
                    'spatial': 0.3,
                    'edge': 0.3
                }
            }
        }
        return PosePipeline(config=config)
        
    @pytest.fixture
    def test_frame(self):
        """创建测试帧"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # 添加一些特征
        cv2.circle(frame, (320, 240), 50, (255, 255, 255), -1)
        cv2.rectangle(frame, (280, 200), (360, 280), (128, 128, 128), -1)
        return frame
        
    def test_basic_process(self, pipeline, test_frame):
        """测试基本处理流程"""
        result = pipeline.process_frame(test_frame)
        
        assert result is not None
        assert result.shape == test_frame.shape
        assert not np.array_equal(result, test_frame)
        
    def test_sequence_process(self, pipeline):
        """测试连续帧处理"""
        frames = []
        results = []
        
        # 生成测试序列
        for i in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # 移动的圆形
            cv2.circle(frame, 
                      (320 + i*10, 240), 
                      50, 
                      (255, 255, 255), 
                      -1)
            frames.append(frame)
            
        # 处理序列
        for frame in frames:
            result = pipeline.process_frame(frame)
            assert result is not None
            results.append(result)
            
        # 验证平滑性
        diffs = []
        for i in range(1, len(results)):
            diff = np.mean(np.abs(
                results[i].astype(float) - 
                results[i-1].astype(float)
            ))
            diffs.append(diff)
            
        avg_diff = np.mean(diffs)
        assert avg_diff < 30.0, "变化不够平滑"
        
    def test_error_handling(self, pipeline):
        """测试错误处理"""
        # 空帧
        result = pipeline.process_frame(None)
        assert result is None
        
        # 无效尺寸
        tiny_frame = np.zeros((1, 1, 3), dtype=np.uint8)
        result = pipeline.process_frame(tiny_frame)
        assert result is None
        
    def test_quality_adaptation(self, pipeline, test_frame):
        """测试质量自适应"""
        # 处理多帧以触发质量评估
        initial_weight = pipeline.smoother.temporal_weight
        
        for _ in range(5):
            # 添加噪声以降低质量
            noisy_frame = test_frame + np.random.normal(0, 30, test_frame.shape)
            noisy_frame = np.clip(noisy_frame, 0, 255).astype(np.uint8)
            
            result = pipeline.process_frame(noisy_frame)
            assert result is not None
            
        # 验证参数自适应
        assert pipeline.smoother.temporal_weight != initial_weight
        
    def test_resource_management(self, pipeline, test_frame):
        """测试资源管理"""
        # 处理一些帧
        for _ in range(3):
            pipeline.process_frame(test_frame)
            
        # 重置
        pipeline.reset()
        assert pipeline._regions == {}
        
        # 再次处理
        result = pipeline.process_frame(test_frame)
        assert result is not None
        
        # 释放资源
        pipeline.release() 
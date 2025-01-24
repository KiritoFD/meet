import pytest
import numpy as np
from pose.processor.jitsi_processor import JitsiPoseProcessor
from pose.types import PoseData

class TestJitsiPoseProcessor:
    @pytest.fixture
    def setup_processor(self):
        """初始化处理器"""
        config = {
            'compression_level': 6,
            'cache_size': 100,
            'optimize_threshold': 0.1
        }
        return JitsiPoseProcessor(config)
        
    def test_process_pose_data(self, setup_processor):
        """测试姿态数据处理"""
        pose_data = PoseData(
            landmarks=[
                {'x': 0.5, 'y': 0.5, 'z': 0.0, 'visibility': 1.0}
                for _ in range(33)  # MediaPipe标准关键点数量
            ],
            timestamp=123.456
        )
        
        processed = setup_processor.process(pose_data)
        assert isinstance(processed, bytes)
        assert len(processed) < len(str(pose_data.landmarks))  # 验证压缩效果
        
    def test_decompress_data(self, setup_processor):
        """测试数据解压"""
        original_data = PoseData(
            landmarks=[{'x': 0.5, 'y': 0.5}],
            timestamp=123.456
        )
        processed = setup_processor.process(original_data)
        decompressed = setup_processor.decompress(processed)
        
        assert decompressed.timestamp == original_data.timestamp
        assert len(decompressed.landmarks) == len(original_data.landmarks)

    def test_landmark_compression(self, setup_processor):
        landmarks = [
            {'x': 0.5, 'y': 0.5, 'z': 0.0, 'visibility': 1.0}
            for _ in range(33)
        ]
        pose_data = PoseData(landmarks=landmarks, timestamp=123.456)
        
        compressed = setup_processor.compress_landmarks(landmarks)
        decompressed = setup_processor.decompress_landmarks(compressed)
        
        # 验证压缩效果和精度
        assert len(compressed) < len(str(landmarks))
        for orig, decomp in zip(landmarks, decompressed):
            assert abs(orig['x'] - decomp['x']) < 0.001
            assert abs(orig['y'] - decomp['y']) < 0.001

    def test_delta_encoding(self, setup_processor):
        # 测试相邻帧的增量编码
        frame1 = PoseData(
            landmarks=[{'x': 0.5, 'y': 0.5}],
            timestamp=1.0
        )
        frame2 = PoseData(
            landmarks=[{'x': 0.51, 'y': 0.52}],
            timestamp=1.1
        )
        
        encoded = setup_processor.encode_delta(frame1, frame2)
        decoded = setup_processor.decode_delta(frame1, encoded)
        
        assert abs(frame2.landmarks[0]['x'] - decoded.landmarks[0]['x']) < 0.001

    def test_cache_mechanism(self, setup_processor):
        # 测试缓存机制
        pose_data = PoseData(
            landmarks=[{'x': 0.5, 'y': 0.5}],
            timestamp=1.0
        )
        
        # 第一次处理
        result1 = setup_processor.process(pose_data)
        
        # 应该使用缓存
        result2 = setup_processor.process(pose_data)
        assert setup_processor._cache_hits > 0 
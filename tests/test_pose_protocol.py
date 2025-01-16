import pytest
import time
import numpy as np
from connect.pose_protocol import PoseProtocol, PoseData

class TestPoseProtocol:
    @pytest.fixture
    def setup_protocol(self):
        """初始化测试环境"""
        return PoseProtocol()
        
    def test_data_encoding(self, setup_protocol):
        """测试数据编码"""
        protocol = setup_protocol
        
        # 创建测试数据
        test_data = PoseData(
            pose_landmarks=[
                {'x': 0.5, 'y': 0.5, 'z': 0.0, 'visibility': 1.0}
            ],
            timestamp=time.time()
        )
        
        # 测试编解码
        encoded = protocol.encode(test_data)
        decoded = protocol.decode(encoded)
        
        assert decoded.pose_landmarks == test_data.pose_landmarks
        assert abs(decoded.timestamp - test_data.timestamp) < 0.001
        
    def test_compression_efficiency(self, setup_protocol):
        """测试压缩效率"""
        protocol = setup_protocol
        
        # 创建大量测试数据
        landmarks = [
            {'x': 0.5, 'y': 0.5, 'z': 0.0, 'visibility': 1.0}
        ] * 100
        
        test_data = PoseData(
            pose_landmarks=landmarks,
            timestamp=time.time()
        )
        
        # 测试压缩率
        encoded = protocol.encode(test_data)
        compression_ratio = len(encoded) / len(str(test_data.pose_landmarks))
        assert compression_ratio < 0.5  # 压缩率>50%
        
    def test_performance(self, setup_protocol):
        """测试性能"""
        protocol = setup_protocol
        test_data = PoseData(
            pose_landmarks=[
                {'x': 0.5, 'y': 0.5, 'z': 0.0, 'visibility': 1.0}
            ] * 33,  # MediaPipe姿态点数量
            timestamp=time.time()
        )
        
        # 测试编码时间
        start_time = time.time()
        for _ in range(100):
            encoded = protocol.encode(test_data)
        encode_time = (time.time() - start_time) / 100
        
        assert encode_time < 0.005  # 5ms内完成编码 
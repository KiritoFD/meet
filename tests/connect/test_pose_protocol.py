import pytest
import numpy as np
import json
import zlib
import time
from connect.pose_protocol import PoseProtocol, PoseData
from connect.errors import InvalidDataError

class TestPoseProtocol:
    @pytest.fixture
    def setup_protocol(self):
        """初始化测试环境"""
        return PoseProtocol(compression_level=6)

    def test_data_encoding(self, setup_protocol):
        """测试数据编码"""
        # 创建测试数据
        pose_data = PoseData(
            pose_landmarks=[
                {'x': 0.5, 'y': 0.5, 'z': 0.0, 'visibility': 1.0}
                for _ in range(33)
            ],
            timestamp=time.time()
        )

        # 编码
        encoded = setup_protocol.encode(pose_data)
        assert isinstance(encoded, bytes)
        
        # 解码
        decoded = setup_protocol.decode(encoded)
        assert isinstance(decoded, PoseData)
        assert len(decoded.pose_landmarks) == 33

    def test_compression_efficiency(self, setup_protocol):
        """测试压缩效率"""
        # 创建大量数据
        pose_data = PoseData(
            pose_landmarks=[
                {'x': np.random.random(), 'y': np.random.random(), 
                 'z': np.random.random(), 'visibility': 1.0}
                for _ in range(100)
            ],
            timestamp=time.time()
        )

        # 压缩
        compressed = setup_protocol.compress(pose_data)
        
        # 验证压缩率
        compression_ratio = len(compressed) / len(str(pose_data.__dict__))
        assert compression_ratio < 0.5  # 压缩率应该大于50%

    def test_error_handling(self, setup_protocol):
        """测试错误处理"""
        # 无效数据
        with pytest.raises(InvalidDataError):
            setup_protocol.encode(None)
        
        with pytest.raises(InvalidDataError):
            setup_protocol.decode(b'invalid data')

        # 格式错误
        with pytest.raises(InvalidDataError):
            setup_protocol.encode({'invalid': 'format'})

    def test_performance(self, setup_protocol):
        """测试性能"""
        pose_data = PoseData(
            pose_landmarks=[
                {'x': np.random.random(), 'y': np.random.random(),
                 'z': np.random.random(), 'visibility': 1.0}
                for _ in range(33)
            ],
            timestamp=time.time()
        )

        # 测试编码性能
        start_time = time.time()
        for _ in range(1000):
            setup_protocol.encode(pose_data)
        encode_time = time.time() - start_time
        assert encode_time < 1.0  # 1000次编码应在1秒内完成 

    def test_protocol_version_compatibility(self, setup_protocol):
        """测试协议版本兼容性"""
        # 测试当前版本
        pose_data = self._create_test_pose_data()
        encoded = setup_protocol.encode(pose_data)
        assert setup_protocol.decode(encoded).version == setup_protocol.CURRENT_VERSION
        
        # 测试向后兼容
        old_data = {
            'version': '1.0',
            'pose_landmarks': pose_data.pose_landmarks,
            'timestamp': pose_data.timestamp
        }
        decoded = setup_protocol.decode_legacy(old_data)
        assert isinstance(decoded, PoseData)

    def test_large_data_handling(self, setup_protocol):
        """测试大数据包处理"""
        # 创建大量姿态点
        large_pose_data = PoseData(
            pose_landmarks=[
                {'x': np.random.random(), 'y': np.random.random(),
                 'z': np.random.random(), 'visibility': 1.0}
                for _ in range(1000)  # 1000个姿态点
            ],
            timestamp=time.time()
        )
        
        # 测试编码和解码
        encoded = setup_protocol.encode(large_pose_data)
        decoded = setup_protocol.decode(encoded)
        assert len(decoded.pose_landmarks) == 1000

    def test_data_validation(self, setup_protocol):
        """测试数据验证"""
        # 测试缺失字段
        invalid_data = {'partial': 'data'}
        with pytest.raises(InvalidDataError):
            setup_protocol.validate(invalid_data)
        
        # 测试无效坐标
        invalid_pose = {
            'landmarks': [{'x': 'invalid', 'y': 0, 'z': 0}]
        }
        with pytest.raises(InvalidDataError):
            setup_protocol.validate(invalid_pose) 
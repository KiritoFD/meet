import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import json
import numpy as np

from connect.jitsi.processor import JitsiPoseProcessor
from lib.jitsi import JitsiError

@pytest.fixture
def config():
    return {
        'processor': {
            'compression_level': 6,
            'max_pose_size': 1024 * 1024,  # 1MB
            'batch_size': 10,
            'processing_timeout': 5.0
        }
    }

@pytest.fixture
def mock_pose_data():
    return {
        'keypoints': np.random.rand(17, 3),  # 17个关键点,每个点有x,y,confidence
        'bbox': [10, 20, 100, 200],  # x, y, width, height
        'score': 0.95,
        'timestamp': 1234567890
    }

@pytest.fixture
def processor(config):
    return JitsiPoseProcessor(config)

class TestJitsiPoseProcessor:
    def test_compress_decompress(self, processor, mock_pose_data):
        compressed = processor.compress(mock_pose_data)
        decompressed = processor.decompress(compressed)
        
        np.testing.assert_array_almost_equal(
            mock_pose_data['keypoints'],
            decompressed['keypoints']
        )
        assert mock_pose_data['bbox'] == decompressed['bbox']

    def test_compress_batch(self, processor, mock_pose_data):
        """测试批量压缩"""
        batch = [mock_pose_data] * 5
        compressed = processor.compress_batch(batch)
        decompressed = processor.decompress_batch(compressed)
        
        assert len(decompressed) == len(batch)
        for orig, dec in zip(batch, decompressed):
            np.testing.assert_array_almost_equal(
                orig['keypoints'],
                dec['keypoints']
            )

    def test_size_limit(self, processor, mock_pose_data):
        """测试大小限制"""
        # 创建超大姿态数据
        large_pose = mock_pose_data.copy()
        large_pose['keypoints'] = np.random.rand(1000, 3)  # 1000个关键点
        
        with pytest.raises(ValueError) as exc_info:
            processor.compress(large_pose)
        assert "exceeds size limit" in str(exc_info.value)

    def test_invalid_data(self, processor):
        """测试无效数据"""
        invalid_poses = [
            None,
            {},
            {'keypoints': []},
            {'keypoints': np.random.rand(17, 2)},  # 缺少confidence
            {'keypoints': np.random.rand(17, 3), 'bbox': [1, 2]},  # bbox不完整
        ]
        
        for pose in invalid_poses:
            with pytest.raises(ValueError):
                processor.compress(pose)

    @pytest.mark.asyncio
    async def test_async_processing(self, processor, mock_pose_data):
        """测试异步处理"""
        async def process_pose(pose):
            await asyncio.sleep(0.1)  # 模拟处理延迟
            return processor.compress(pose)
            
        tasks = [process_pose(mock_pose_data) for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        for compressed in results:
            decompressed = processor.decompress(compressed)
            np.testing.assert_array_almost_equal(
                mock_pose_data['keypoints'],
                decompressed['keypoints']
            )

    def test_compression_level(self, processor, mock_pose_data):
        """测试不同压缩级别"""
        # 使用不同压缩级别
        sizes = []
        for level in range(1, 10):
            processor.config['processor']['compression_level'] = level
            compressed = processor.compress(mock_pose_data)
            sizes.append(len(compressed))
            
        # 验证压缩级别越高,大小越小
        assert all(sizes[i] >= sizes[i+1] for i in range(len(sizes)-1))

    def test_batch_size_limit(self, processor, mock_pose_data):
        """测试批处理大小限制"""
        batch_size = processor.config['processor']['batch_size']
        large_batch = [mock_pose_data] * (batch_size + 1)
        
        with pytest.raises(ValueError) as exc_info:
            processor.compress_batch(large_batch)
        assert "batch size" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_processing_timeout(self, processor, mock_pose_data):
        """测试处理超时"""
        processor.config['processor']['processing_timeout'] = 0.1
        
        async def slow_process():
            await asyncio.sleep(0.2)  # 超过超时时间
            return processor.compress(mock_pose_data)
            
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                slow_process(),
                timeout=processor.config['processor']['processing_timeout']
            )

    def test_data_validation(self, processor):
        """测试数据验证"""
        valid_pose = {
            'keypoints': np.random.rand(17, 3),
            'bbox': [0, 0, 100, 100],
            'score': 0.9,
            'timestamp': 123456
        }
        assert processor.validate_pose(valid_pose)
        
        invalid_poses = [
            {'keypoints': np.random.rand(16, 3)},  # 关键点数量错误
            {'keypoints': np.random.rand(17, 3), 'score': 1.5},  # 分数超范围
            {'keypoints': np.random.rand(17, 3), 'bbox': [-1, 0, 100, 100]},  # 负坐标
        ]
        
        for pose in invalid_poses:
            assert not processor.validate_pose(pose)

    def test_compression_stability(self, processor, mock_pose_data):
        """测试压缩稳定性"""
        # 多次压缩相同数据应该得到相同结果
        results = [
            processor.compress(mock_pose_data)
            for _ in range(10)
        ]
        
        # 验证所有结果相同
        assert all(r == results[0] for r in results)

    def test_error_handling(self, processor):
        """测试错误处理"""
        # 测试压缩错误
        with patch.object(processor, '_compress_data', side_effect=Exception("Compression failed")):
            with pytest.raises(JitsiError):
                processor.compress({'keypoints': np.random.rand(17, 3)})
                
        # 测试解压错误
        with pytest.raises(JitsiError):
            processor.decompress(b"invalid data")

    def test_performance_metrics(self, processor, mock_pose_data):
        """测试性能指标"""
        # 收集压缩性能指标
        compressed = processor.compress(mock_pose_data)
        metrics = processor.get_metrics()
        
        assert 'compression_ratio' in metrics
        assert 'processing_time' in metrics
        assert metrics['compression_ratio'] > 0
        assert metrics['processing_time'] > 0 
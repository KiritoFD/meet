import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import json
import numpy as np
import time

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
        """测试基本的压缩和解压功能"""
        compressed = processor.compress(mock_pose_data)
        decompressed = processor.decompress(compressed)
        
        # 验证关键点数据
        np.testing.assert_array_almost_equal(
            mock_pose_data['keypoints'],
            decompressed['keypoints']
        )
        # 验证其他字段
        assert mock_pose_data['bbox'] == decompressed['bbox']
        assert abs(mock_pose_data['score'] - decompressed['score']) < 1e-6
        assert mock_pose_data['timestamp'] == decompressed['timestamp']

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
        """测试无效数据处理"""
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

    def test_compression_ratio(self, processor, mock_pose_data):
        """测试压缩比"""
        compressed = processor.compress(mock_pose_data)
        original_size = len(json.dumps(mock_pose_data).encode())
        compressed_size = len(compressed)
        
        ratio = compressed_size / original_size
        assert ratio < 1.0  # 确保有压缩效果
        assert ratio > 0.1  # 确保压缩不过度

    def test_processing_timeout(self, processor, mock_pose_data):
        """测试处理超时"""
        with patch.object(processor, '_compress_data') as mock_compress:
            mock_compress.side_effect = lambda x: asyncio.sleep(6.0)  # 超过超时时间
            
            with pytest.raises(asyncio.TimeoutError):
                processor.compress(mock_pose_data)

    def test_batch_size_limit(self, processor, mock_pose_data):
        """测试批处理大小限制"""
        oversized_batch = [mock_pose_data] * (processor.config['batch_size'] + 1)
        with pytest.raises(ValueError) as exc_info:
            processor.compress_batch(oversized_batch)
        assert "batch size limit" in str(exc_info.value)

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

    def test_concurrent_processing(self, processor, mock_pose_data):
        """测试并发处理"""
        async def concurrent_test():
            tasks = []
            for i in range(10):
                data = mock_pose_data.copy()
                data['timestamp'] = i
                tasks.append(asyncio.create_task(
                    processor.async_compress(data)
                ))
            results = await asyncio.gather(*tasks)
            assert len(results) == 10
            
            # 验证结果
            for i, compressed in enumerate(results):
                decompressed = processor.decompress(compressed)
                assert decompressed['timestamp'] == i
                
        asyncio.run(concurrent_test())

    def test_memory_usage(self, processor, mock_pose_data):
        """测试内存使用"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # 进行大量压缩操作
        for _ in range(1000):
            processor.compress(mock_pose_data)
            
        # 检查内存增长是否在合理范围
        current_memory = process.memory_info().rss
        memory_increase = (current_memory - initial_memory) / 1024 / 1024  # MB
        assert memory_increase < 100  # 内存增长不应超过100MB 

    def test_data_validation(self, processor):
        """测试数据验证"""
        # 测试边界值
        edge_cases = [
            {'keypoints': np.zeros((17, 3)), 'score': 0.0},  # 最小值
            {'keypoints': np.ones((17, 3)), 'score': 1.0},   # 最大值
            {'keypoints': np.random.rand(17, 3), 'score': 0.5, 
             'bbox': [0, 0, 1, 1]},  # 最小bbox
        ]
        for case in edge_cases:
            processor.validate_pose(case)  # 不应抛出异常

    def test_compression_stability(self, processor, mock_pose_data):
        """测试压缩稳定性"""
        # 修复：bytes不能直接用set()，需要转换为字符串比较
        results = []
        for _ in range(5):
            compressed = processor.compress(mock_pose_data)
            results.append(compressed)
        # 比较所有结果是否相同
        assert all(r == results[0] for r in results)

    def test_compression_levels(self, processor, mock_pose_data):
        """测试不同压缩级别"""
        sizes = []
        for level in range(1, 10):
            processor.config['compression_level'] = level
            compressed = processor.compress(mock_pose_data)
            sizes.append(len(compressed))
        
        # 验证压缩级别越高，压缩率越好
        assert all(sizes[i] >= sizes[i+1] for i in range(len(sizes)-1))

    def test_recovery_from_errors(self, processor, mock_pose_data):
        """测试错误恢复"""
        # 模拟临时失败后的恢复
        with patch.object(processor, '_compress_data') as mock_compress:
            mock_compress.side_effect = [
                Exception("Temporary failure"),
                processor._compress_data(mock_pose_data)
            ]
            
            # 第一次应该失败
            with pytest.raises(JitsiError):
                processor.compress(mock_pose_data)
                
            # 第二次应该成功
            compressed = processor.compress(mock_pose_data)
            assert compressed is not None

    def test_resource_cleanup(self, processor):
        """测试资源清理"""
        # 模拟资源使用
        processor.start_processing()
        try:
            processor.process_data(b"test data")
        finally:
            processor.cleanup()
        
        # 验证资源已释放
        assert processor.is_clean()

    @pytest.mark.asyncio
    async def test_cancellation(self, processor, mock_pose_data):
        """测试任务取消"""
        async def slow_process():
            await asyncio.sleep(1.0)
            return processor.compress(mock_pose_data)

        task = asyncio.create_task(slow_process())
        await asyncio.sleep(0.1)
        task.cancel()
        
        with pytest.raises(asyncio.CancelledError):
            await task 

    def test_data_serialization(self, processor, mock_pose_data):
        """测试数据序列化格式"""
        compressed = processor.compress(mock_pose_data)
        # 验证序列化格式
        header = compressed[:8]  # 假设前8字节是头部
        assert header.startswith(b'POSE')  # 检查魔数
        assert len(compressed) >= 8  # 确保至少包含头部

    def test_incremental_processing(self, processor):
        """测试增量处理"""
        data = []
        for i in range(5):
            processor.process_chunk(f"chunk_{i}".encode())
            data.append(processor.get_processed_data())
        assert len(data) == 5
        assert all(d is not None for d in data)

    @pytest.mark.asyncio
    async def test_backpressure(self, processor, mock_pose_data):
        """测试背压机制"""
        # 模拟处理速度慢于输入速度的情况
        processor.set_processing_delay(0.1)  # 每帧处理延迟0.1秒
        
        async def producer():
            for _ in range(10):
                await processor.queue_data(mock_pose_data)
                
        async def consumer():
            while processor.has_pending_data():
                await processor.process_next()
                
        # 并发运行生产者和消费者
        await asyncio.gather(producer(), consumer())
        assert processor.queue_size() == 0

    @pytest.mark.asyncio
    async def test_concurrent_load(self, processor, mock_pose_data):
        """测试并发负载"""
        # 修复：添加错误处理和超时控制
        async def worker():
            try:
                for _ in range(100):
                    await asyncio.wait_for(
                        processor.async_compress(mock_pose_data),
                        timeout=1.0
                    )
            except asyncio.TimeoutError:
                pytest.fail("Processing timeout")
                
        try:
            workers = [worker() for _ in range(10)]
            await asyncio.gather(*workers)
        except Exception as e:
            pytest.fail(f"Concurrent processing failed: {e}")

    def test_config_validation(self):
        """测试配置验证"""
        # 无效配置
        invalid_configs = [
            {'processor': {'compression_level': 0}},  # 压缩级别太低
            {'processor': {'compression_level': 11}},  # 压缩级别太高
            {'processor': {'max_pose_size': -1}},     # 负的大小限制
            {'processor': {'batch_size': 0}},         # 无效批大小
        ]
        
        for config in invalid_configs:
            with pytest.raises(ValueError):
                JitsiPoseProcessor(config)

@pytest.mark.benchmark
class TestProcessorPerformance:
    """性能测试"""
    
    def test_throughput(self, processor, mock_pose_data):
        """测试吞吐量"""
        start_time = time.time()
        count = 0
        while time.time() - start_time < 1.0:  # 测试1秒
            processor.compress(mock_pose_data)
            count += 1
        assert count > 100  # 每秒至少处理100帧

    def test_latency_distribution(self, processor, mock_pose_data):
        """测试延迟分布"""
        latencies = []
        for _ in range(100):
            start = time.time()
            processor.compress(mock_pose_data)
            latencies.append(time.time() - start)
            
        # 分析延迟分布
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        
        assert p50 < 0.01  # 中位数延迟<10ms
        assert p95 < 0.05  # 95%延迟<50ms
        assert p99 < 0.1   # 99%延迟<100ms

    def test_memory_leak(self, processor, mock_pose_data):
        """测试内存泄漏"""
        import gc
        gc.collect()  # 强制垃圾回收
        
        initial_objects = len(gc.get_objects())
        for _ in range(1000):
            processor.compress(mock_pose_data)
            processor.clear_metrics()  # 清理指标数据
            
        gc.collect()
        final_objects = len(gc.get_objects())
        # 对象数量增长不应过大
        assert final_objects - initial_objects < 100

    def test_cpu_usage(self, processor, mock_pose_data):
        """测试CPU使用"""
        import psutil
        process = psutil.Process()
        
        # 记录初始CPU时间
        initial_cpu_times = process.cpu_times()
        
        # 执行密集处理
        for _ in range(1000):
            processor.compress(mock_pose_data)
            
        # 记录最终CPU时间
        final_cpu_times = process.cpu_times()
        
        # 计算CPU使用率
        user_time = final_cpu_times.user - initial_cpu_times.user
        sys_time = final_cpu_times.system - initial_cpu_times.system
        total_time = user_time + sys_time
        
        # CPU时间不应过高
        assert total_time < 5.0  # 不超过5秒CPU时间

@pytest.mark.stress
class TestProcessorStress:
    """压力测试"""
    
    async def test_concurrent_load(self, processor, mock_pose_data):
        """测试并发负载"""
        async def worker():
            for _ in range(100):
                await processor.async_compress(mock_pose_data)
                
        workers = [worker() for _ in range(10)]
        await asyncio.gather(*workers) 

@pytest.mark.cv
def test_image_processing(cv2, processor, mock_pose_data):
    """测试图像处理功能"""
    # 只有当cv2可用时才会运行这个测试
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = processor.process_image(image)
    assert result is not None 
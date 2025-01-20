import pytest
import time
import asyncio
from unittest.mock import Mock

class TestUnifiedSender:
    @pytest.mark.asyncio
    async def test_latency(self, unified_sender):
        """测试端到端延迟"""
        pose_data = generate_test_pose()
        
        latencies = []
        for _ in range(100):
            start = time.time()
            success = await unified_sender.send(
                data_type='pose',
                data=pose_data
            )
            latency = time.time() - start
            assert success
            latencies.append(latency)
            
        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 0.05  # 50ms延迟要求

    @pytest.mark.asyncio
    async def test_bandwidth_control(self, unified_sender):
        """测试带宽控制"""
        unified_sender.set_bandwidth_limit(1_000_000)  # 1MB/s
        large_data = generate_test_pose(landmark_count=1000)
        
        start_time = time.time()
        sent_bytes = 0
        
        for _ in range(100):
            success = await unified_sender.send(
                data_type='pose',
                data=large_data
            )
            assert success
            sent_bytes += len(str(large_data))
            
        duration = time.time() - start_time
        bandwidth_usage = sent_bytes / duration
        assert bandwidth_usage <= 1_000_000  # 不超过限制

    @pytest.mark.asyncio
    async def test_priority_queue(self, unified_sender):
        """测试优先级队列"""
        # 填满队列
        unified_sender.queue = asyncio.Queue(maxsize=5)
        
        # 发送不同优先级的数据
        high_priority = generate_test_pose()
        low_priority = generate_test_pose()
        
        await unified_sender.send(data_type='pose', data=low_priority, priority=0)
        await unified_sender.send(data_type='pose', data=high_priority, priority=9)
        
        # 验证高优先级数据先发送
        sent_data = unified_sender._socket.emit.call_args_list
        assert len(sent_data) >= 2
        assert sent_data[1][0][1]['priority'] == 9

    def test_data_validation(self, unified_sender):
        """测试数据验证"""
        invalid_data = {'invalid': 'data'}
        with pytest.raises(ValueError):
            asyncio.run(unified_sender.send(
                data_type='pose',
                data=invalid_data
            )) 
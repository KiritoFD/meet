import pytest
from connect.jitsi.transport import JitsiTransport
import asyncio

@pytest.mark.benchmark
@pytest.mark.parametrize("message_size, batch_size", [
    (1024, 1),    # 小消息单条发送
    (10240, 10),  # 中等消息批量发送
    (102400, 5)   # 大消息批量发送
])
async def test_throughput(benchmark, message_size, batch_size):
    """测试不同消息大小的吞吐量"""
    transport = JitsiTransport({'batch_size': batch_size})
    transport._data_channel = AsyncMock()
    
    # 生成测试数据
    test_data = [b"x" * message_size for _ in range(1000)]
    
    # 执行基准测试
    result = await benchmark(transport.batch_send, test_data)
    
    # 验证性能指标
    assert result.count(True) == len(test_data)
    assert benchmark.stats.stats.mean < 0.1  # 平均延迟小于100ms 
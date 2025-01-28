import pytest
from connect.jitsi.transport import JitsiTransport
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_network_recovery():
    """测试网络中断恢复"""
    transport = JitsiTransport({})
    transport._data_channel = AsyncMock()
    
    # 模拟首次连接失败
    transport._jitsi.connect = AsyncMock(side_effect=ConnectionError)
    assert await transport.connect("test_room") is False
    
    # 第二次连接成功
    transport._jitsi.connect = AsyncMock(return_value=AsyncMock())
    assert await transport.connect("test_room") is True

@pytest.mark.asyncio
async def test_retry_mechanism():
    """测试数据传输重试机制"""
    transport = JitsiTransport({'retry_limit': 3})
    transport._data_channel = AsyncMock()
    
    # 模拟前两次发送失败
    transport._data_channel.send = AsyncMock(
        side_effect=[TimeoutError, TimeoutError, True]
    )
    
    result = await transport.send(b"test")
    assert result is True
    assert transport._data_channel.send.call_count == 3 
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import json

from connect.jitsi.client import JitsiClient
from lib.jitsi import JitsiError

@pytest.fixture
def config():
    return {
        'jitsi': {
            'host': 'test.jitsi.com',
            'port': 443,
            'ice_servers': [
                {'urls': ['stun:stun.l.google.com:19302']}
            ]
        },
        'client': {
            'reconnect_attempts': 3,
            'reconnect_delay': 1.0,
            'buffer_size': 100
        }
    }

@pytest.fixture
def mock_transport():
    mock = AsyncMock()
    mock.connect = AsyncMock()
    mock.send = AsyncMock()
    mock.close = AsyncMock()
    mock.get_stats = Mock(return_value={
        'connection_state': 'connected',
        'sent': 0,
        'errors': 0,
        'latency': []
    })
    return mock

@pytest.fixture
async def client(config, mock_transport):
    with patch('connect.jitsi.client.JitsiTransport', return_value=mock_transport):
        client = JitsiClient(config)
        yield client
        await client.close()

class TestJitsiClient:
    @pytest.mark.asyncio
    async def test_connect_success(self, client, mock_transport):
        """测试连接成功"""
        await client.connect("test_room")
        
        mock_transport.connect.assert_called_once_with("test_room")
        assert client.is_connected()
        assert client.room_id == "test_room"

    @pytest.mark.asyncio
    async def test_connect_failure(self, client, mock_transport):
        """测试连接失败"""
        mock_transport.connect.side_effect = JitsiError("Connection failed")
        
        with pytest.raises(JitsiError):
            await client.connect("test_room")
            
        assert not client.is_connected()
        assert client.room_id is None

    @pytest.mark.asyncio
    async def test_reconnect(self, client, mock_transport):
        """测试重连机制"""
        await client.connect("test_room")
        
        # 模拟连接断开
        mock_transport.get_stats.return_value['connection_state'] = 'closed'
        mock_transport.send.side_effect = [JitsiError(), None]  # 第一次失败,第二次成功
        
        # 发送数据应该触发重连
        await client.send(b"test")
        
        assert mock_transport.connect.call_count == 2
        assert client.is_connected()

    @pytest.mark.asyncio
    async def test_send_data(self, client, mock_transport):
        """测试发送数据"""
        await client.connect("test_room")
        
        test_data = b"test_message"
        await client.send(test_data)
        
        mock_transport.send.assert_called_once_with(test_data)

    @pytest.mark.asyncio
    async def test_batch_send(self, client, mock_transport):
        """测试批量发送"""
        await client.connect("test_room")
        
        messages = [b"msg1", b"msg2", b"msg3"]
        await client.batch_send(messages)
        
        assert mock_transport.send.call_count == len(messages)

    @pytest.mark.asyncio
    async def test_data_handler(self, client, mock_transport):
        """测试数据处理器"""
        received_data = []
        
        @client.on_data
        async def handler(data: bytes):
            received_data.append(data)
            
        await client.connect("test_room")
        
        # 模拟接收数据
        data_handler = mock_transport.on_data.call_args[0][0]
        await data_handler(b"test_data")
        
        assert received_data == [b"test_data"]

    @pytest.mark.asyncio
    async def test_error_handling(self, client, mock_transport):
        """测试错误处理"""
        await client.connect("test_room")
        
        # 模拟发送错误
        mock_transport.send.side_effect = JitsiError("Send failed")
        
        with pytest.raises(JitsiError):
            await client.send(b"test")
            
        stats = client.get_stats()
        assert stats['errors'] > 0

    @pytest.mark.asyncio
    async def test_buffer_overflow(self, client, mock_transport):
        """测试缓冲区溢出"""
        await client.connect("test_room")
        
        # 填充超过缓冲区大小的数据
        buffer_size = client.config['client']['buffer_size']
        data = [b"test"] * (buffer_size + 10)
        
        await client.batch_send(data)
        assert mock_transport.send.call_count == buffer_size

    @pytest.mark.asyncio
    async def test_close(self, client, mock_transport):
        """测试关闭客户端"""
        await client.connect("test_room")
        await client.close()
        
        mock_transport.close.assert_called_once()
        assert not client.is_connected()
        assert client.room_id is None

    @pytest.mark.asyncio
    async def test_stats(self, client, mock_transport):
        """测试统计信息"""
        await client.connect("test_room")
        
        mock_transport.get_stats.return_value.update({
            'sent': 10,
            'errors': 2,
            'latency': [0.1, 0.2, 0.3]
        })
        
        stats = client.get_stats()
        assert stats['sent'] == 10
        assert stats['errors'] == 2
        assert len(stats['latency']) == 3

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, client, mock_transport):
        """测试并发操作"""
        await client.connect("test_room")
        
        # 并发发送多条消息
        tasks = [
            client.send(f"msg{i}".encode())
            for i in range(10)
        ]
        await asyncio.gather(*tasks)
        
        assert mock_transport.send.call_count == 10

    @pytest.mark.asyncio
    async def test_connection_state_changes(self, client, mock_transport):
        """测试连接状态变化"""
        states = []
        
        @client.on_state_change
        def state_handler(state: str):
            states.append(state)
            
        await client.connect("test_room")
        mock_transport.get_stats.return_value['connection_state'] = 'closed'
        await client.close()
        
        assert 'connected' in states
        assert 'closed' in states

    @pytest.mark.asyncio
    async def test_connect_with_jitsi_meet(self, client, mock_jitsi):
        await client.connect("test_room")
        mock_jitsi.connect.assert_called_once()
        assert client.is_connected()

    @pytest.mark.asyncio
    async def test_reconnect_with_jitsi_meet(self, client, mock_jitsi):
        mock_jitsi.connect.side_effect = [JitsiError(), None]
        await client.connect("test_room")
        assert mock_jitsi.connect.call_count == 2 
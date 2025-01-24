import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List
import time
import random
import psutil

from connect.jitsi.transport import JitsiTransport, TransportError
from lib.jitsi import (
    JitsiMeet,
    JitsiConnection,
    JitsiConference,
    JitsiDataChannel
)

class MockDataChannel:
    def __init__(self):
        self.is_ready = True
        self.messages = []
        self.on_message = None
        self.on_open = None
        self.on_close = None
        
    def ready(self) -> bool:
        return self.is_ready
        
    async def send(self, data: bytes) -> None:
        if not self.is_ready:
            raise ConnectionError("Channel not ready")
        self.messages.append(data)
        
    async def close(self) -> None:
        self.is_ready = False
        if self.on_close:
            self.on_close()
            
    def get_state(self) -> str:
        return 'open' if self.is_ready else 'closed'

class MockConference:
    def __init__(self):
        self.data_channels = {}
        
    async def create_data_channel(self, label: str, options: Dict) -> MockDataChannel:
        channel = MockDataChannel()
        self.data_channels[label] = channel
        return channel
        
    async def leave(self) -> None:
        for channel in self.data_channels.values():
            await channel.close()

@pytest.fixture
def config() -> Dict:
    return {
        'jitsi_host': 'test.jitsi.com',
        'jitsi_port': 443,
        'buffer_size': 30,
        'batch_size': 10,
        'conference': {
            'room_id': 'test_room',
            'data_channel_options': {
                'ordered': True,
                'maxRetransmits': 3
            }
        }
    }

@pytest.fixture
async def transport(config):
    """初始化传输层"""
    transport = JitsiTransport(config)
    yield transport
    await transport.close()

@pytest.fixture
def mock_jitsi():
    """Mock Jitsi 组件"""
    with patch('lib.jitsi.JitsiMeet') as mock:
        mock.return_value.connect = AsyncMock(return_value=Mock())
        mock.return_value.connect.return_value.join_conference = AsyncMock(
            return_value=MockConference()
        )
        yield mock

class TestJitsiTransport:
    class TestConnection:
        @pytest.mark.asyncio
        async def test_connect_success(self, transport, mock_jitsi, config):
            """测试连接成功"""
            success = await transport.connect("test_room")
            assert success
            assert transport._data_channel is not None
            assert transport._data_channel.ready()
            mock_jitsi.assert_called_once_with(
                host=config['jitsi_host'],
                port=config['jitsi_port']
            )

        @pytest.mark.asyncio
        async def test_connect_failure(self, transport, mock_jitsi):
            """测试连接失败"""
            mock_jitsi.return_value.connect.side_effect = ConnectionError()
            
            with pytest.raises(TransportError) as exc_info:
                await transport.connect("test_room")
            assert "Failed to connect" in str(exc_info.value)

    class TestDataTransfer:
        @pytest.mark.asyncio
        async def test_send_data_success(self, transport, mock_jitsi):
            """测试数据发送成功"""
            await transport.connect("test_room")
            test_data = b"test_payload"
            
            success = await transport.send(test_data)
            assert success
            assert transport._stats['sent'] == 1
            assert len(transport._stats['latency']) == 1
            assert transport._data_channel.messages == [test_data]

        @pytest.mark.asyncio
        async def test_send_data_with_reconnect(self, transport, mock_jitsi):
            """测试发送时自动重连"""
            await transport.connect("test_room")
            transport._data_channel.is_ready = False
            
            test_data = b"test_after_reconnect"
            success = await transport.send(test_data)
            
            assert success
            assert transport._data_channel.ready()
            assert transport._data_channel.messages == [test_data]

        @pytest.mark.asyncio
        async def test_send_data_failure(self, transport, mock_jitsi):
            """测试发送失败"""
            await transport.connect("test_room")
            transport._data_channel.send = AsyncMock(side_effect=ConnectionError())
            
            with pytest.raises(TransportError) as exc_info:
                await transport.send(b"test_data")
            assert transport._stats['errors'] == 1

        @pytest.mark.asyncio
        async def test_batch_send(self, transport, mock_jitsi):
            """测试批量发送"""
            await transport.connect("test_room")
            data_list = [f"data_{i}".encode() for i in range(15)]
            
            results = await transport.batch_send(data_list)
            assert len(results) == 15
            assert all(results)
            assert transport._stats['sent'] == 15
            assert len(transport._data_channel.messages) == 15

        @pytest.mark.asyncio
        async def test_batch_send_chunking(self, transport, mock_jitsi):
            """测试大批量数据分块发送"""
            await transport.connect("test_room")
            data_list = [f"data_{i}".encode() for i in range(25)]  # > batch_size
            
            results = await transport.batch_send(data_list)
            assert len(results) == 25
            assert all(results)
            assert transport._stats['sent'] == 25

        @pytest.mark.asyncio
        async def test_data_handlers(self, transport, mock_jitsi):
            """测试数据处理回调"""
            received_data = []
            
            @transport.on_data
            async def handler(data: bytes):
                received_data.append(data)
            
            await transport.connect("test_room")
            test_data = b"test_message"
            await transport._handle_message(test_data)
            
            assert received_data == [test_data]
            assert transport._stats['received'] == 1

        @pytest.mark.asyncio
        async def test_handler_error(self, transport, mock_jitsi):
            """测试处理器错误"""
            @transport.on_data
            async def failing_handler(data: bytes):
                raise ValueError("Handler error")
            
            await transport.connect("test_room")
            await transport._handle_message(b"test")
            assert transport._stats['errors'] == 1

    class TestErrorHandling:
        @pytest.mark.asyncio
        async def test_cleanup(self, transport, mock_jitsi):
            """测试资源清理"""
            await transport.connect("test_room")
            assert transport._data_channel is not None
            
            await transport.close()
            assert transport._data_channel is None
            assert transport._conference is None
            assert transport._connection is None
            assert len(transport._buffer) == 0

        @pytest.mark.asyncio
        async def test_stats_collection(self, transport, mock_jitsi):
            """测试统计信息收集"""
            await transport.connect("test_room")
            
            # 发送一些数据
            for _ in range(5):
                await transport.send(b"test")
            
            stats = transport.get_stats()
            assert stats['sent'] == 5
            assert stats['received'] == 0
            assert stats['errors'] == 0
            assert 'avg_latency' in stats
            assert stats['connection_state'] == 'open'

        @pytest.mark.asyncio
        async def test_state_transitions(self, transport, mock_jitsi):
            """测试所有可能的状态转换"""
            # 初始状态
            assert transport.get_stats()['connection_state'] == 'closed'
            
            # 连接中
            connect_task = asyncio.create_task(transport.connect("test_room"))
            await asyncio.sleep(0)  # 让出控制权
            assert transport.get_stats()['connection_state'] == 'connecting'
            
            # 连接完成
            await connect_task
            assert transport.get_stats()['connection_state'] == 'open'
            
            # 模拟断开
            transport._data_channel.is_ready = False
            assert transport.get_stats()['connection_state'] == 'closed'
            
            # 重连
            await transport.send(b"test")  # 触发重连
            assert transport.get_stats()['connection_state'] == 'open'

        @pytest.mark.asyncio
        async def test_buffer_management(self, transport, mock_jitsi):
            """测试缓冲区管理"""
            await transport.connect("test_room")
            
            # 填充缓冲区
            for i in range(40):  # > buffer_size
                transport._buffer.append(f"data_{i}".encode())
            
            assert len(transport._buffer) == transport.config['buffer_size']
            assert transport._buffer[0] == b"data_10"  # 验证FIFO行为 

        @pytest.mark.asyncio
        async def test_connect_with_custom_options(self, transport, mock_jitsi, config):
            """测试使用自定义选项连接"""
            custom_options = {'timeout': 30, 'reconnect_attempts': 3}
            success = await transport.connect("test_room", options=custom_options)
            assert success
            assert transport._connection is not None

        @pytest.mark.asyncio
        async def test_connect_timeout(self, transport, mock_jitsi):
            """测试连接超时"""
            mock_jitsi.return_value.connect = AsyncMock(side_effect=asyncio.TimeoutError)
            with pytest.raises(TransportError) as exc_info:
                await transport.connect("test_room")
            assert "timeout" in str(exc_info.value).lower()

        @pytest.mark.asyncio
        async def test_auto_reconnect(self, transport, mock_jitsi):
            """测试自动重连机制"""
            await transport.connect("test_room")
            # 模拟连接断开
            transport._data_channel.is_ready = False
            # 模拟重连延迟
            await asyncio.sleep(0.1)
            # 发送数据触发重连
            success = await transport.send(b"test")
            assert success
            assert transport._data_channel.ready()

        @pytest.mark.asyncio
        async def test_send_large_data(self, transport, mock_jitsi):
            """测试发送大数据"""
            await transport.connect("test_room")
            large_data = b"x" * (1024 * 1024)  # 1MB
            success = await transport.send(large_data)
            assert success
            assert len(transport._data_channel.messages) == 1

        @pytest.mark.asyncio
        async def test_send_empty_data(self, transport, mock_jitsi):
            """测试发送空数据"""
            await transport.connect("test_room")
            with pytest.raises(ValueError):
                await transport.send(b"")

        @pytest.mark.asyncio
        async def test_send_rate_limiting(self, transport, mock_jitsi):
            """测试发送速率限制"""
            await transport.connect("test_room")
            start_time = time.time()
            data = b"test"
            for _ in range(100):  # 快速发送100条消息
                await transport.send(data)
            duration = time.time() - start_time
            assert duration >= 0.1  # 确保有基本的速率限制

        @pytest.mark.asyncio
        async def test_batch_send_empty_list(self, transport, mock_jitsi):
            """测试批量发送空列表"""
            await transport.connect("test_room")
            results = await transport.batch_send([])
            assert results == []

        @pytest.mark.asyncio
        async def test_batch_send_mixed_sizes(self, transport, mock_jitsi):
            """测试批量发送不同大小的数据"""
            await transport.connect("test_room")
            data_list = [
                b"small",
                b"x" * 1000,  # 1KB
                b"y" * 10000,  # 10KB
            ]
            results = await transport.batch_send(data_list)
            assert all(results)
            assert len(transport._data_channel.messages) == 3

        @pytest.mark.asyncio
        async def test_batch_send_partial_failure(self, transport, mock_jitsi):
            """测试批量发送部分失败"""
            await transport.connect("test_room")
            # 模拟间歇性失败
            original_send = transport._data_channel.send
            fail_count = 0
            async def mock_send(data):
                nonlocal fail_count
                fail_count += 1
                if fail_count % 3 == 0:
                    raise ConnectionError()
                await original_send(data)
            transport._data_channel.send = mock_send
            
            data_list = [b"test"] * 10
            results = await transport.batch_send(data_list)
            assert len([r for r in results if r]) >= 7  # 至少70%成功

        @pytest.mark.asyncio
        async def test_handle_invalid_state(self, transport):
            """测试无效状态下的操作"""
            with pytest.raises(TransportError):
                await transport.send(b"test")  # 未连接就发送

        @pytest.mark.asyncio
        async def test_handle_connection_drop(self, transport, mock_jitsi):
            """测试连接断开处理"""
            await transport.connect("test_room")
            transport._connection = None
            with pytest.raises(TransportError):
                await transport.send(b"test")

        @pytest.mark.asyncio
        async def test_handle_conference_error(self, transport, mock_jitsi):
            """测试会议错误处理"""
            mock_jitsi.return_value.connect.return_value.join_conference.side_effect = \
                Exception("Conference error")
            with pytest.raises(TransportError):
                await transport.connect("test_room")

        @pytest.mark.asyncio
        async def test_concurrent_sends(self, transport, mock_jitsi):
            """测试并发发送"""
            await transport.connect("test_room")
            tasks = []
            for i in range(50):  # 并发50个发送任务
                tasks.append(transport.send(f"data_{i}".encode()))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = len([r for r in results if not isinstance(r, Exception)])
            assert success_count >= 45  # 90%成功率

        @pytest.mark.asyncio
        async def test_long_running_stability(self, transport, mock_jitsi):
            """测试长时间运行稳定性"""
            await transport.connect("test_room")
            start_time = time.time()
            error_count = 0
            
            for _ in range(100):  # 模拟100次操作
                try:
                    await transport.send(b"test")
                    await asyncio.sleep(0.01)  # 模拟实际操作间隔
                except Exception:
                    error_count += 1
                
            duration = time.time() - start_time
            assert error_count <= 5  # 允许5%的错误率
            assert duration >= 1.0  # 确保不会运行太快

        @pytest.mark.asyncio
        async def test_memory_usage(self, transport, mock_jitsi):
            """测试内存使用"""
            await transport.connect("test_room")
            initial_size = len(transport._buffer)
            
            # 模拟大量数据处理
            for _ in range(1000):
                transport._buffer.append(b"test")
            
            assert len(transport._buffer) <= transport.config['buffer_size']
            await transport.close()
            assert len(transport._buffer) == 0

        @pytest.mark.asyncio
        async def test_resource_cleanup_after_error(self, transport, mock_jitsi):
            """测试错误后的资源清理"""
            await transport.connect("test_room")
            transport._data_channel.send = AsyncMock(side_effect=Exception("Forced error"))
            
            try:
                await transport.send(b"test")
            except:
                pass
            
            await transport.close()
            assert transport._data_channel is None
            assert transport._connection is None

        @pytest.mark.asyncio
        async def test_detailed_stats(self, transport, mock_jitsi):
            """测试详细统计信息"""
            await transport.connect("test_room")
            
            # 模拟各种操作
            await transport.send(b"success")
            try:
                transport._data_channel.send = AsyncMock(side_effect=Exception)
                await transport.send(b"fail")
            except:
                pass
            
            stats = transport.get_stats()
            assert stats['sent'] == 1
            assert stats['errors'] == 1
            assert 'avg_latency' in stats
            assert 'connection_state' in stats

        @pytest.mark.asyncio
        async def test_performance_metrics(self, transport, mock_jitsi):
            """测试性能指标收集"""
            await transport.connect("test_room")
            
            # 模拟不同大小的数据发送
            sizes = [10, 100, 1000, 10000]
            for size in sizes:
                await transport.send(b"x" * size)
            
            stats = transport.get_stats()
            assert len(stats['latency']) == len(sizes)
            assert 0 < stats['avg_latency'] < 1.0  # 合理的延迟范围

        @pytest.mark.asyncio
        async def test_reconnect_with_backoff(self, transport, mock_jitsi):
            """测试重连退避机制"""
            await transport.connect("test_room")
            transport._data_channel.is_ready = False
            
            start_time = time.time()
            for _ in range(3):
                try:
                    await transport.send(b"test")
                except:
                    pass
            
            duration = time.time() - start_time
            assert duration >= 0.3  # 验证退避延迟

        @pytest.mark.asyncio
        async def test_handle_rapid_connect_disconnect(self, transport, mock_jitsi):
            """测试快速连接断开"""
            for _ in range(5):
                await transport.connect("test_room")
                await transport.close()
            
            # 最终状态检查
            assert transport._data_channel is None
            assert transport._connection is None
            assert len(transport._buffer) == 0 

        @pytest.mark.asyncio
        async def test_connect_with_invalid_room(self, transport, mock_jitsi):
            """测试无效房间ID"""
            invalid_rooms = ["", " ", None]
            for room_id in invalid_rooms:
                with pytest.raises(TransportError) as exc_info:
                    await transport.connect(room_id)
                assert "invalid room id" in str(exc_info.value).lower()

        @pytest.mark.asyncio
        async def test_send_oversized_data(self, transport, mock_jitsi):
            """测试超大数据发送"""
            await transport.connect("test_room")
            max_size = transport.config.get('max_message_size', 5 * 1024 * 1024)  # 默认5MB
            oversized_data = b"x" * (max_size + 1024)  # 超过限制
            
            with pytest.raises(TransportError) as exc_info:
                await transport.send(oversized_data)
            assert "data too large" in str(exc_info.value).lower()

        @pytest.mark.asyncio
        async def test_send_invalid_data_type(self, transport, mock_jitsi):
            """测试无效数据类型"""
            await transport.connect("test_room")
            invalid_data = [
                "string data",  # 字符串
                123,  # 整数
                {"key": "value"},  # 字典
                [1, 2, 3]  # 列表
            ]
            
            for data in invalid_data:
                with pytest.raises((TypeError, TransportError)):
                    await transport.send(data)

        def test_invalid_config(self):
            """测试无效配置"""
            invalid_configs = [
                {},  # 空配置
                {'jitsi_host': ''},  # 无效主机
                {'jitsi_host': 'test.com', 'buffer_size': -1},  # 无效缓冲区
                {'jitsi_host': 'test.com', 'batch_size': 0},  # 无效批量大小
                {'jitsi_host': 'test.com', 'conference': None}  # 无效会议配置
            ]
            
            for config in invalid_configs:
                with pytest.raises(ValueError) as exc_info:
                    JitsiTransport(config)
                assert any(x in str(exc_info.value).lower() 
                          for x in ['invalid', 'missing', 'required'])

        def test_config_defaults(self):
            """测试配置默认值"""
            minimal_config = {
                'jitsi_host': 'test.jitsi.com',
                'conference': {'room_id': 'test_room'}
            }
            
            transport = JitsiTransport(minimal_config)
            assert transport.config['buffer_size'] > 0
            assert transport.config['batch_size'] > 0
            assert isinstance(transport.config['conference']['data_channel_options'], dict)

        @pytest.mark.asyncio
        async def test_multiple_data_handlers(self, transport, mock_jitsi):
            """测试多个数据处理器"""
            handler_results = {
                'handler1': [],
                'handler2': [],
                'handler3': []
            }
            
            @transport.on_data
            async def handler1(data: bytes):
                handler_results['handler1'].append(data)
            
            @transport.on_data
            async def handler2(data: bytes):
                handler_results['handler2'].append(data)
                await asyncio.sleep(0.1)  # 模拟耗时操作
            
            @transport.on_data
            async def handler3(data: bytes):
                handler_results['handler3'].append(data)
            
            await transport.connect("test_room")
            test_data = b"test_multi_handler"
            await transport._handle_message(test_data)
            
            # 验证所有处理器都被调用
            for handler_data in handler_results.values():
                assert handler_data == [test_data]

        @pytest.mark.asyncio
        async def test_handler_exception_isolation(self, transport, mock_jitsi):
            """测试处理器异常隔离"""
            handler_called = {
                'handler1': False,
                'handler2': False,
                'handler3': False
            }
            
            @transport.on_data
            async def handler1(data: bytes):
                handler_called['handler1'] = True
                raise ValueError("Handler 1 error")
            
            @transport.on_data
            async def handler2(data: bytes):
                handler_called['handler2'] = True
                raise RuntimeError("Handler 2 error")
            
            @transport.on_data
            async def handler3(data: bytes):
                handler_called['handler3'] = True
            
            await transport.connect("test_room")
            await transport._handle_message(b"test")
            
            # 验证所有处理器都被调用，即使有错误
            assert all(handler_called.values())
            assert transport._stats['errors'] == 2

        @pytest.mark.asyncio
        async def test_concurrent_connect_requests(self, transport, mock_jitsi):
            """测试并发连接请求"""
            connect_results = []
            
            async def connect_attempt():
                try:
                    result = await transport.connect("test_room")
                    connect_results.append(result)
                except TransportError:
                    connect_results.append(False)
                
            # 并发发起10个连接请求
            tasks = [connect_attempt() for _ in range(10)]
            await asyncio.gather(*tasks)
            
            # 应该只有一个成功的连接
            assert sum(connect_results) == 1
            assert transport._connection is not None

        @pytest.mark.asyncio
        async def test_concurrent_buffer_access(self, transport, mock_jitsi):
            """测试并发缓冲区访问"""
            await transport.connect("test_room")
            buffer_size = transport.config['buffer_size']
            
            async def buffer_writer():
                for i in range(100):
                    transport._buffer.append(f"data_{i}".encode())
                    await asyncio.sleep(0.001)
                
            async def buffer_reader():
                for _ in range(100):
                    buffer_len = len(transport._buffer)
                    assert buffer_len <= buffer_size
                    await asyncio.sleep(0.001)
                
            await asyncio.gather(
                buffer_writer(),
                buffer_reader(),
                buffer_writer(),
                buffer_reader()
            )
            
            assert len(transport._buffer) <= buffer_size

        @pytest.mark.asyncio
        async def test_no_resource_leak_after_errors(self, transport, mock_jitsi):
            """测试错误后无资源泄漏"""
            import gc
            import weakref
            
            # 记录初始引用
            gc.collect()
            initial_refs = len(gc.get_referrers(transport))
            data_channels = []
            
            # 模拟多次错误场景
            for _ in range(10):
                try:
                    await transport.connect("test_room")
                    if transport._data_channel:
                        data_channels.append(weakref.ref(transport._data_channel))
                    transport._data_channel.send = AsyncMock(
                        side_effect=Exception("Forced error")
                    )
                    await transport.send(b"test")
                except:
                    await transport.close()
                
            # 检查资源清理
            gc.collect()
            final_refs = len(gc.get_referrers(transport))
            assert final_refs <= initial_refs
            
            # 检查通道对象是否被正确清理
            assert all(ref() is None for ref in data_channels)

        @pytest.mark.asyncio
        async def test_cleanup_on_context_exit(self, transport, mock_jitsi):
            """测试上下文退出时的清理"""
            import gc
            import weakref
            
            await transport.connect("test_room")
            
            # 创建弱引用
            channel_ref = weakref.ref(transport._data_channel)
            conference_ref = weakref.ref(transport._conference)
            connection_ref = weakref.ref(transport._connection)
            
            # 关闭传输
            await transport.close()
            gc.collect()
            
            # 验证所有组件都被清理
            assert channel_ref() is None
            assert conference_ref() is None
            assert connection_ref() is None
            assert len(transport._buffer) == 0

        @pytest.mark.asyncio
        async def test_performance_degradation(self, transport, mock_jitsi):
            """测试长时间运行的性能退化"""
            await transport.connect("test_room")
            
            # 记录初始性能
            latencies = []
            for _ in range(10):  # 取平均值
                start_time = time.time()
                await transport.send(b"test")
                latencies.append(time.time() - start_time)
            initial_latency = sum(latencies) / len(latencies)
            
            # 模拟大量操作
            for i in range(1000):
                transport._buffer.append(f"data_{i}".encode())
                await transport.send(b"test")
                if i % 100 == 0:
                    await asyncio.sleep(0.01)  # 防止过度占用CPU
                
            # 检查最终性能
            latencies = []
            for _ in range(10):
                start_time = time.time()
                await transport.send(b"test")
                latencies.append(time.time() - start_time)
            final_latency = sum(latencies) / len(latencies)
            
            # 允许最多100%的性能退化
            assert final_latency <= initial_latency * 2
            
        @pytest.mark.asyncio
        async def test_memory_growth(self, transport, mock_jitsi):
            """测试内存增长"""
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss
            
            await transport.connect("test_room")
            
            # 模拟大量数据处理
            for _ in range(10000):
                await transport.send(b"test" * 100)
                transport._buffer.append(b"test" * 100)
            
            current_memory = process.memory_info().rss
            memory_growth = (current_memory - initial_memory) / initial_memory
            
            # 内存增长不应超过50%
            assert memory_growth <= 0.5

        @pytest.mark.asyncio
        async def test_message_ordering(self, transport, mock_jitsi):
            """测试消息顺序性"""
            await transport.connect("test_room")
            messages = []
            
            @transport.on_data
            async def order_checker(data: bytes):
                messages.append(data)
            
            # 发送带序号的消息
            for i in range(100):
                await transport.send(str(i).encode())
            
            # 验证消息顺序
            assert len(messages) == 100
            for i, msg in enumerate(messages):
                assert int(msg.decode()) == i

        @pytest.mark.asyncio
        async def test_message_deduplication(self, transport, mock_jitsi):
            """测试消息去重"""
            await transport.connect("test_room")
            received = set()
            
            @transport.on_data
            async def dedup_checker(data: bytes):
                received.add(data)
            
            # 重复发送相同消息
            test_data = b"duplicate_message"
            for _ in range(5):
                await transport.send(test_data)
            
            assert len(received) == 1

        @pytest.mark.asyncio
        async def test_network_latency_handling(self, transport, mock_jitsi):
            """测试网络延迟处理"""
            await transport.connect("test_room")
            
            # 模拟网络延迟
            original_send = transport._data_channel.send
            async def delayed_send(data):
                await asyncio.sleep(0.1)  # 100ms延迟
                await original_send(data)
            transport._data_channel.send = delayed_send
            
            start_time = time.time()
            await transport.send(b"test")
            duration = time.time() - start_time
            
            assert 0.1 <= duration <= 0.2  # 允许0.1s误差
            assert transport._stats['latency'][-1] >= 0.1

        @pytest.mark.asyncio
        async def test_network_jitter_handling(self, transport, mock_jitsi):
            """测试网络抖动处理"""
            await transport.connect("test_room")
            
            # 模拟不稳定的网络延迟
            original_send = transport._data_channel.send
            async def jittery_send(data):
                jitter = random.uniform(0.01, 0.2)  # 10-200ms随机延迟
                await asyncio.sleep(jitter)
                await original_send(data)
            transport._data_channel.send = jittery_send
            
            latencies = []
            for _ in range(10):
                start_time = time.time()
                await transport.send(b"test")
                latencies.append(time.time() - start_time)
            
            # 验证延迟波动在预期范围内
            assert 0.01 <= min(latencies) <= 0.2
            assert max(latencies) - min(latencies) <= 0.19  # 最大抖动

        @pytest.mark.asyncio
        async def test_flow_control(self, transport, mock_jitsi):
            """测试流量控制"""
            await transport.connect("test_room")
            
            # 模拟背压
            backpressure = asyncio.Event()
            original_send = transport._data_channel.send
            async def controlled_send(data):
                await backpressure.wait()  # 等待释放背压
                await original_send(data)
            transport._data_channel.send = controlled_send
            
            # 启动发送任务
            send_task = asyncio.create_task(transport.send(b"test"))
            await asyncio.sleep(0.1)
            assert not send_task.done()  # 应该被阻塞
            
            # 释放背压
            backpressure.set()
            await send_task
            assert send_task.done()

        @pytest.mark.asyncio
        async def test_buffer_overflow_handling(self, transport, mock_jitsi):
            """测试缓冲区溢出处理"""
            await transport.connect("test_room")
            buffer_size = transport.config['buffer_size']
            
            # 快速填充缓冲区
            for i in range(buffer_size * 2):
                transport._buffer.append(b"test")
            
            assert len(transport._buffer) == buffer_size
            # 验证是FIFO行为
            assert transport._buffer[0] == b"test"
            assert transport._buffer[-1] == b"test"

        @pytest.mark.asyncio
        async def test_recovery_after_transport_error(self, transport, mock_jitsi):
            """测试传输错误后的恢复"""
            await transport.connect("test_room")
            
            # 模拟传输错误
            transport._data_channel.send = AsyncMock(
                side_effect=[ConnectionError(), ConnectionError(), None]
            )
            
            # 第一次尝试应该失败
            with pytest.raises(TransportError):
                await transport.send(b"test1")
            
            # 重置mock
            transport._data_channel.send = AsyncMock()
            
            # 应该能够恢复正常发送
            success = await transport.send(b"test2")
            assert success

        @pytest.mark.asyncio
        async def test_partial_connection_recovery(self, transport, mock_jitsi):
            """测试部分连接恢复"""
            await transport.connect("test_room")
            
            # 模拟连接部分断开
            transport._connection = None
            assert transport._data_channel is not None
            
            # 应该能自动修复连接
            success = await transport.send(b"test")
            assert success
            assert transport._connection is not None

        @pytest.mark.asyncio
        async def test_max_message_rate(self, transport, mock_jitsi):
            """测试最大消息速率"""
            await transport.connect("test_room")
            start_time = time.time()
            
            # 尝试快速发送大量消息
            messages = [b"test"] * 1000
            results = await transport.batch_send(messages)
            
            duration = time.time() - start_time
            message_rate = len(results) / duration
            
            # 验证消息速率不超过限制
            assert message_rate <= transport.config.get('max_message_rate', float('inf'))

        @pytest.mark.asyncio
        async def test_resource_limits(self, transport, mock_jitsi):
            """测试资源限制"""
            await transport.connect("test_room")
            
            # 测试内存限制
            large_data = b"x" * (1024 * 1024)  # 1MB
            memory_before = psutil.Process().memory_info().rss
            
            for _ in range(10):
                await transport.send(large_data)
            
            memory_after = psutil.Process().memory_info().rss
            memory_growth = (memory_after - memory_before) / (1024 * 1024)  # MB
            
            # 确保内存增长在合理范围
            assert memory_growth < 100  # 不应该增长超过100MB

        @pytest.mark.asyncio
        async def test_resource_contention(self, transport, mock_jitsi):
            """测试资源竞争"""
            await transport.connect("test_room")
            
            # 模拟资源竞争
            async def buffer_manipulator():
                for _ in range(100):
                    transport._buffer.append(b"test")
                    await asyncio.sleep(0.001)
                    transport._buffer.pop()
                
            async def stats_reader():
                for _ in range(100):
                    _ = transport.get_stats()
                    await asyncio.sleep(0.001)
                
            async def sender():
                for _ in range(100):
                    await transport.send(b"test")
                    await asyncio.sleep(0.001)
            
            # 并发执行多个资源操作
            await asyncio.gather(
                buffer_manipulator(),
                stats_reader(),
                sender(),
                buffer_manipulator(),
                stats_reader()
            )
            
            # 验证最终状态
            assert transport._data_channel.ready()
            assert len(transport._buffer) <= transport.config['buffer_size']

        @pytest.mark.benchmark
        async def test_performance_benchmark(self, transport, mock_jitsi, benchmark):
            """性能基准测试"""
            await transport.connect("test_room")
            
            async def benchmark_send():
                for _ in range(100):
                    await transport.send(b"benchmark")
            
            # 运行基准测试
            result = await benchmark(benchmark_send)
            
            # 验证性能指标
            assert result.mean < 0.01  # 平均延迟小于10ms
            assert result.stddev < 0.005  # 标准差小于5ms

        @pytest.mark.asyncio
        async def test_state_consistency(self, transport, mock_jitsi):
            """测试状态一致性"""
            states = []
            
            async def state_checker():
                while len(states) < 100:
                    states.append(transport.get_stats()['connection_state'])
                    await asyncio.sleep(0.01)
            
            # 启动状态检查器
            checker = asyncio.create_task(state_checker())
            
            # 执行一系列操作
            await transport.connect("test_room")
            await transport.send(b"test")
            await transport.close()
            await transport.connect("test_room")
            await transport.close()
            
            await checker
            
            # 验证状态转换的合法性
            valid_transitions = {
                'closed': {'connecting'},
                'connecting': {'open', 'closed'},
                'open': {'closed'}
            }
            
            for i in range(len(states) - 1):
                assert states[i+1] in valid_transitions[states[i]]
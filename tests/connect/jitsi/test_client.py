import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from connect.jitsi.client import JitsiClient
from connect.jitsi.errors import JitsiError, ConnectionError, MediaError

@pytest.fixture
def config():
    return {
        'client': {
            'host': 'test.jitsi.com',
            'port': 443,
            'retry_attempts': 3,
            'retry_delay': 1.0,
            'timeout': 30.0,
            'ice_servers': [
                {'urls': ['stun:stun.l.google.com:19302']}
            ]
        }
    }

@pytest.fixture
async def client(config):
    client = JitsiClient(config)
    yield client
    await client.disconnect()

class TestJitsiClient:
    @pytest.mark.asyncio
    async def test_connect(self, client):
        """测试连接功能"""
        with patch.object(client, '_establish_connection', new_callable=AsyncMock) as mock_connect:
            await client.connect()
            mock_connect.assert_called_once()
            assert client.is_connected

    @pytest.mark.asyncio
    async def test_disconnect(self, client):
        """测试断开连接"""
        await client.connect()
        await client.disconnect()
        assert not client.is_connected

    @pytest.mark.asyncio
    async def test_reconnect(self, client):
        """测试重连功能"""
        await client.connect()
        
        # 模拟连接断开
        client._connection.close()
        assert not client.is_connected
        
        # 测试自动重连
        await client.ensure_connected()
        assert client.is_connected

    @pytest.mark.asyncio
    async def test_connection_error_handling(self, client):
        """测试连接错误处理"""
        with patch.object(client, '_establish_connection', 
                         side_effect=ConnectionError(code=3001)):
            with pytest.raises(ConnectionError) as exc_info:
                await client.connect()
            assert exc_info.value.code == 3001

    @pytest.mark.asyncio
    async def test_media_handling(self, client):
        """测试媒体处理"""
        await client.connect()
        
        # 测试音频设备切换
        devices = await client.get_available_devices()
        if len(devices['audioinput']) > 1:
            old_device = client.current_audio_device
            new_device = next(d for d in devices['audioinput'] if d != old_device)
            await client.switch_audio_device(new_device)
            assert client.current_audio_device == new_device
            
        # 测试视频分辨率切换
        resolutions = [(640, 480), (1280, 720), (1920, 1080)]
        for width, height in resolutions:
            await client.set_video_resolution(width, height)
            constraints = client.get_media_constraints()
            assert constraints['video']['width'] == width
            assert constraints['video']['height'] == height

    @pytest.mark.asyncio
    async def test_device_enumeration(self, client):
        """测试设备枚举"""
        with patch.object(client, '_get_media_devices') as mock_devices:
            mock_devices.return_value = {
                'audioinput': ['default'],
                'videoinput': ['default'],
                'audiooutput': ['default']
            }
            
            devices = await client.get_available_devices()
            assert 'audioinput' in devices
            assert 'videoinput' in devices
            assert 'audiooutput' in devices

    @pytest.mark.asyncio
    async def test_device_selection(self, client):
        """测试设备选择"""
        # 测试选择音频设备
        await client.select_audio_device('test_audio')
        assert client.current_audio_device == 'test_audio'
        
        # 测试选择视频设备
        await client.select_video_device('test_video')
        assert client.current_video_device == 'test_video'

    @pytest.mark.asyncio
    async def test_media_constraints(self, client):
        """测试媒体约束"""
        constraints = {
            'audio': {'echoCancellation': True},
            'video': {'width': 1280, 'height': 720}
        }
        
        await client.set_media_constraints(constraints)
        current = client.get_media_constraints()
        assert current['audio']['echoCancellation'] is True
        assert current['video']['width'] == 1280

    @pytest.mark.asyncio
    async def test_connection_quality(self, client):
        """测试连接质量监控"""
        # 模拟连接质量数据
        stats = {
            'bitrate': 1000,
            'packetLoss': 0.1,
            'jitter': 20,
            'rtt': 50
        }
        
        with patch.object(client, '_get_connection_stats', 
                         return_value=stats):
            quality = await client.get_connection_quality()
            assert quality['bitrate'] == 1000
            assert quality['quality_score'] > 0

    @pytest.mark.asyncio
    async def test_error_recovery(self, client):
        """测试错误恢复"""
        # 模拟连续失败然后恢复
        connection_results = [
            ConnectionError(code=3001),  # 第一次失败
            ConnectionError(code=3001),  # 第二次失败
            None  # 第三次成功
        ]
        
        with patch.object(client, '_establish_connection', 
                         side_effect=connection_results):
            await client.connect()  # 应该最终成功
            assert client.is_connected
            assert client.retry_count == 2

    @pytest.mark.asyncio
    async def test_resource_cleanup(self, client):
        """测试资源清理"""
        await client.connect()
        
        # 获取初始资源状态
        initial_resources = client.get_resource_usage()
        
        # 执行一些操作
        await client.enable_audio()
        await client.enable_video()
        await client.disable_audio()
        await client.disable_video()
        
        # 断开连接
        await client.disconnect()
        
        # 验证资源已清理
        final_resources = client.get_resource_usage()
        assert final_resources['active_tracks'] == 0
        assert final_resources['peer_connections'] == 0

    @pytest.mark.asyncio
    async def test_event_handling(self, client):
        """测试事件处理"""
        events = []
        
        @client.on('connected')
        def handle_connect():
            events.append('connected')
            
        @client.on('disconnected')
        def handle_disconnect():
            events.append('disconnected')
            
        await client.connect()
        await client.disconnect()
        
        assert events == ['connected', 'disconnected']

    @pytest.mark.asyncio
    async def test_connection_lifecycle(self, client):
        """测试连接生命周期"""
        # 测试初始状态
        assert not client.is_connected
        assert client.state == 'disconnected'
        
        # 连接过程
        await client.connect()
        assert client.is_connected
        assert client.state == 'connected'
        
        # 暂停
        await client.pause()
        assert client.state == 'paused'
        
        # 恢复
        await client.resume()
        assert client.state == 'connected'
        
        # 断开
        await client.disconnect()
        assert not client.is_connected
        assert client.state == 'disconnected'

    @pytest.mark.asyncio
    async def test_media_stream_handling(self, client):
        """测试媒体流处理"""
        await client.connect()
        
        # 测试音频流
        audio_stream = await client.create_audio_stream()
        assert audio_stream.active
        await client.stop_audio_stream()
        assert not audio_stream.active
        
        # 测试视频流
        video_stream = await client.create_video_stream()
        assert video_stream.active
        await client.stop_video_stream()
        assert not video_stream.active

    @pytest.mark.asyncio
    async def test_data_channel(self, client):
        """测试数据通道"""
        await client.connect()
        
        # 创建数据通道
        channel = await client.create_data_channel('test')
        assert channel.label == 'test'
        assert channel.readyState == 'open'
        
        # 发送数据
        test_data = {'type': 'test', 'content': 'hello'}
        await channel.send(test_data)
        
        # 关闭通道
        await channel.close()
        assert channel.readyState == 'closed'

    @pytest.mark.asyncio
    async def test_error_handling_comprehensive(self, client):
        """全面的错误处理测试"""
        error_scenarios = [
            # 连接错误
            (client._establish_connection, ConnectionError(code=3001), "连接建立失败"),
            (client._establish_connection, asyncio.TimeoutError(), "连接超时"),
            # 媒体错误
            (client.enable_audio, MediaError(code=4002), "媒体权限被拒绝"),
            (client.enable_video, MediaError(code=4001), "设备未找到"),
            # ICE错误
            (client._add_ice_candidate, ValueError("Invalid candidate"), "无效的ICE候选"),
            # 信令错误
            (client.send_signaling, ConnectionError(code=3003), "信令发送失败"),
            # 新增更多错误场景
            (client._create_offer, MediaError(code=4003), "媒体流创建失败"),
            (client._set_remote_description, ValueError("Invalid SDP"), "无效的SDP"),
            (client.create_data_channel, ConnectionError(code=3006), "数据通道创建失败")
        ]
        
        for method, error, description in error_scenarios:
            with pytest.raises(Exception) as exc_info:
                with patch.object(client, method.__name__, side_effect=error):
                    await method()
            assert str(exc_info.value) == description
            # 验证错误恢复
            if isinstance(error, ConnectionError):
                assert client._recovery_attempts > 0

    def test_config_validation(self):
        """测试配置验证"""
        # 测试必需字段
        with pytest.raises(ValueError):
            JitsiClient({})
        
        # 测试无效的ICE服务器
        invalid_config = {
            'client': {
                'ice_servers': 'invalid'
            }
        }
        with pytest.raises(ValueError):
            JitsiClient(invalid_config)
        
        # 测试无效的超时值
        invalid_config = {
            'client': {
                'timeout': -1
            }
        }
        with pytest.raises(ValueError):
            JitsiClient(invalid_config)

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_connection_performance(self, client):
        """测试连接性能"""
        import time
        
        # 测试连接时间
        start = time.time()
        await client.connect()
        connect_time = time.time() - start
        assert connect_time < 1.0  # 连接应在1秒内完成
        
        # 测试媒体切换延迟
        start = time.time()
        await client.enable_video()
        await client.disable_video()
        switch_time = time.time() - start
        assert switch_time < 0.5  # 切换应在0.5秒内完成 

    @pytest.mark.asyncio
    async def test_signaling(self, client):
        """测试信令处理"""
        await client.connect()
        
        # 测试信令消息发送
        message = {
            'type': 'offer',
            'sdp': 'test_sdp'
        }
        await client.send_signaling(message)
        
        # 测试信令消息接收
        received_messages = []
        @client.on('signaling')
        def handle_signaling(msg):
            received_messages.append(msg)
            
        await client._handle_signaling({'type': 'answer', 'sdp': 'remote_sdp'})
        assert len(received_messages) == 1
        assert received_messages[0]['type'] == 'answer'

    @pytest.mark.asyncio
    async def test_ice_handling(self, client):
        """测试ICE处理"""
        await client.connect()
        
        # 测试ICE候选收集
        candidates = []
        @client.on('ice_candidate')
        def handle_ice(candidate):
            candidates.append(candidate)
            
        test_candidate = {
            'candidate': 'candidate:1 1 UDP 2130706431 192.168.1.1 53421 typ host',
            'sdpMLineIndex': 0,
            'sdpMid': '0'
        }
        await client._handle_ice_candidate(test_candidate)
        
        assert len(candidates) == 1
        assert candidates[0]['sdpMid'] == '0'

    @pytest.mark.asyncio
    async def test_stats_collection(self, client):
        """测试统计数据收集"""
        await client.connect()
        
        # 启用统计收集
        await client.enable_stats_collection()
        
        # 生成一些活动
        await client.enable_audio()
        await asyncio.sleep(0.1)
        await client.disable_audio()
        
        # 获取统计数据
        stats = await client.get_stats()
        assert 'audio' in stats
        assert 'timestamp' in stats
        assert stats['audio'].get('packetsLost') is not None

    @pytest.mark.asyncio
    async def test_bandwidth_control(self, client):
        """测试带宽控制"""
        await client.connect()
        
        # 设置带宽限制
        constraints = {
            'video': {
                'maxBitrate': 1000000,  # 1Mbps
                'maxFramerate': 30
            }
        }
        await client.set_bandwidth_constraints(constraints)
        
        # 验证应用的约束
        current = await client.get_bandwidth_info()
        assert current['video']['maxBitrate'] == 1000000
        assert current['video']['maxFramerate'] == 30

    @pytest.mark.asyncio
    async def test_network_adaptation(self, client):
        """测试网络适应性"""
        await client.connect()
        
        # 模拟网络状况变化
        network_conditions = [
            {'bandwidth': 1000000, 'latency': 50},  # 好的网络
            {'bandwidth': 500000, 'latency': 100},  # 一般网络
            {'bandwidth': 100000, 'latency': 200}   # 差的网络
        ]
        
        for condition in network_conditions:
            await client._handle_network_change(condition)
            stats = await client.get_connection_quality()
            assert stats['bandwidth'] == condition['bandwidth']
            assert stats['latency'] == condition['latency']
            # 验证客户端是否相应调整了媒体质量
            media_config = await client.get_current_media_config()
            assert media_config['video']['quality'] in ['high', 'medium', 'low'] 

    @pytest.mark.asyncio
    async def test_security_features(self, client):
        """测试安全特性"""
        # 测试加密
        await client.connect()
        assert client.is_connection_encrypted()
        
        # 测试身份验证
        token = "test_token"
        await client.authenticate(token)
        assert client.is_authenticated
        
        # 测试权限检查
        assert await client.has_permission("share_screen")
        assert not await client.has_permission("admin_operations") 

    @pytest.mark.asyncio
    async def test_peer_connection_states(self, client):
        """测试对等连接状态"""
        await client.connect()
        
        # 测试状态转换
        states = []
        @client.on('peer_connection_state')
        def handle_state(state):
            states.append(state)
            
        await client._handle_peer_connection_state('connecting')
        await client._handle_peer_connection_state('connected')
        await client._handle_peer_connection_state('disconnected')
        
        assert states == ['connecting', 'connected', 'disconnected']

    @pytest.mark.asyncio
    async def test_renegotiation(self, client):
        """测试重新协商"""
        await client.connect()
        
        # 模拟需要重新协商的场景
        await client.enable_video()  # 初始只有视频
        
        # 记录初始SDP
        initial_sdp = client._connection.local_description
        
        # 添加音频，触发重新协商
        await client.enable_audio()
        
        # 验证SDP已更新
        new_sdp = client._connection.local_description
        assert new_sdp != initial_sdp
        assert 'audio' in new_sdp.sdp

    @pytest.mark.asyncio
    async def test_simulcast(self, client):
        """测试多流传输"""
        await client.connect()
        
        # 启用多流传输
        await client.enable_simulcast()
        
        # 验证视频配置
        constraints = client.get_media_constraints()
        assert 'simulcast' in constraints['video']
        assert len(constraints['video']['simulcast']['encodings']) == 3  # 低中高三种质量 

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_performance_comprehensive(self, client):
        """全面的性能测试"""
        # 添加CPU使用率监控
        import psutil
        process = psutil.Process()
        
        # 测试CPU使用
        initial_cpu_percent = process.cpu_percent()
        await asyncio.sleep(0.1)  # 等待CPU数据收集
        
        # 执行密集操作
        for _ in range(100):
            await client.enable_video()
            await client.disable_video()
            
        final_cpu_percent = process.cpu_percent()
        cpu_increase = final_cpu_percent - initial_cpu_percent
        assert cpu_increase < 50  # CPU增长不应超过50%
        
        # 测试并发性能
        async def concurrent_operation():
            await client.enable_video()
            await asyncio.sleep(0.01)
            await client.disable_video()
            
        tasks = [concurrent_operation() for _ in range(10)]
        start_time = time.time()
        await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        assert total_time < 1.0  # 并发操作应在1秒内完成

    @pytest.mark.asyncio
    async def test_cleanup_comprehensive(self, client):
        """全面的清理测试"""
        # 创建各种资源
        await client.connect()
        await client.enable_audio()
        await client.enable_video()
        data_channel = await client.create_data_channel('test')
        
        # 记录资源状态
        resources_before = {
            'peer_connections': len(client._peer_connections),
            'data_channels': len(client._data_channels),
            'media_streams': len(client._media_streams)
        }
        
        # 正常关闭
        await client.disconnect()
        
        # 验证资源释放
        assert len(client._peer_connections) == 0
        assert len(client._data_channels) == 0
        assert len(client._media_streams) == 0
        
        # 验证事件监听器清理
        assert len(client._event_handlers) == 0 

    @pytest.mark.asyncio
    async def test_media_quality_adaptation(self, client):
        """测试媒体质量自适应"""
        await client.connect()
        await client.enable_video()
        
        # 测试不同网络条件下的质量调整
        quality_levels = {
            'high': {'bandwidth': 2000000, 'latency': 30},
            'medium': {'bandwidth': 1000000, 'latency': 100},
            'low': {'bandwidth': 500000, 'latency': 200}
        }
        
        for quality, condition in quality_levels.items():
            await client._handle_network_change(condition)
            await asyncio.sleep(0.1)  # 等待调整
            config = await client.get_current_media_config()
            assert config['video']['quality'] == quality

    @pytest.mark.asyncio
    async def test_connection_recovery_strategies(self, client):
        """测试连接恢复策略"""
        await client.connect()
        
        # 测试不同类型的断开恢复
        scenarios = [
            ('network_lost', ConnectionError(code=3003)),
            ('ice_failed', ConnectionError(code=3004)),
            ('signaling_lost', ConnectionError(code=3005))
        ]
        
        for scenario_type, error in scenarios:
            # 模拟断开
            await client._handle_connection_error(error)
            assert not client.is_connected
            
            # 验证恢复策略
            recovery_method = client._get_recovery_strategy(scenario_type)
            await recovery_method()
            assert client.is_connected

    @pytest.mark.asyncio
    async def test_media_fallback(self, client):
        """测试媒体回退机制"""
        await client.connect()
        
        # 测试视频回退到音频
        await client.enable_video()
        await client._handle_network_change({'bandwidth': 100000})  # 低带宽
        config = await client.get_current_media_config()
        assert not config['video']['enabled']
        assert config['audio']['enabled']
        
        # 测试音频质量降级
        await client._handle_network_change({'bandwidth': 50000})  # 极低带宽
        config = await client.get_current_media_config()
        assert config['audio']['bitrate'] < 32000  # 降至低比特率
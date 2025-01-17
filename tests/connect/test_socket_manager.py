import pytest
from unittest.mock import Mock, patch
import socketio
from connect.socket_manager import SocketManager, ConnectionConfig
from connect.errors import ConnectionError
import time

class TestSocketManager:
    @pytest.fixture
    def mock_socketio(self):
        """创建模拟的SocketIO对象"""
        mock_sio = Mock(spec=socketio.Server)
        return mock_sio
        
    @pytest.fixture
    def mock_audio_processor(self):
        """创建模拟的音频处理器"""
        return Mock()

    @pytest.fixture
    def setup_manager(self, mock_socketio, mock_audio_processor):
        """初始化测试环境"""
        manager = SocketManager(mock_socketio, mock_audio_processor)
        yield manager
        manager.disconnect()

    def test_connection_lifecycle(self, setup_manager):
        """测试连接生命周期"""
        # 连接
        assert setup_manager.connect()
        assert setup_manager.connected

        # 断开
        setup_manager.disconnect()
        assert not setup_manager.connected

    def test_event_handling(self, setup_manager):
        """测试事件处理"""
        test_data = {'message': 'test'}
        received_data = None

        @setup_manager.on('test_event')
        def handle_test(data):
            nonlocal received_data
            received_data = data

        # 模拟接收事件
        setup_manager._handle_event('test_event', test_data)
        assert received_data == test_data

    def test_error_handling(self, setup_manager):
        """测试错误处理"""
        # 模拟发送错误
        setup_manager.sio.emit.side_effect = Exception("Send failed")
        with pytest.raises(ConnectionError):
            setup_manager.emit('test', {})

    def test_reconnection(self, setup_manager):
        """测试重连机制"""
        # 模拟断开连接
        setup_manager.disconnect()
        assert not setup_manager.connected

        # 测试自动重连
        setup_manager.sio.connected = True
        assert setup_manager.connect()
        assert setup_manager.connected

    def test_performance(self, setup_manager):
        """测试性能"""
        setup_manager.connect()

        # 测试发送延迟
        start_time = time.time()
        setup_manager.emit('test', {'data': 'test'})
        latency = time.time() - start_time
        assert latency < 0.001  # 发送延迟应小于1ms

    @patch('socketio.Server')
    def test_concurrent_connections(self, mock_socketio, mock_audio_processor):
        """测试并发连接"""
        connections = []
        for i in range(10):
            manager = SocketManager(mock_socketio, mock_audio_processor)
            assert manager.connect()
            connections.append(manager)
        
        # 验证连接数
        assert SocketManager._active_connections == 10
        
        # 清理连接
        for manager in connections:
            manager.disconnect()
        
        assert SocketManager._active_connections == 0

    @patch('socketio.Server')
    def test_reconnection_with_data_recovery(self, mock_socketio, mock_audio_processor):
        """测试重连和数据恢复"""
        manager = SocketManager(mock_socketio, mock_audio_processor)
        
        # 模拟连接成功
        assert manager.connect()
        
        # 模拟数据发送
        test_data = {"message": "test"}
        manager.emit("test_event", test_data)
        
        # 模拟断开连接
        manager.disconnect()
        
        # 模拟重连
        assert manager.connect()
        
        # 验证重连后的状态
        assert manager.connected
        
        # 清理
        manager.disconnect() 
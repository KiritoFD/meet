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
        mock_sio = Mock(spec=socketio.Client)
        mock_sio.event = lambda f: f  # 模拟装饰器
        return mock_sio

    @pytest.fixture
    def setup_manager(self, mock_socketio):
        """初始化测试环境"""
        config = ConnectionConfig(
            url='http://localhost:5000',
            reconnect_attempts=3,
            reconnect_delay=100
        )
        manager = SocketManager(config, socketio_client=mock_socketio)
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

        # 重连
        assert setup_manager.reconnect()
        assert setup_manager.connected

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

    def test_error_handling(self, setup_manager, mock_socketio):
        """测试错误处理"""
        # 模拟连接错误
        mock_socketio.connect.side_effect = Exception("Connection failed")
        with pytest.raises(ConnectionError):
            setup_manager.connect()

        # 模拟发送错误
        mock_socketio.emit.side_effect = Exception("Send failed")
        with pytest.raises(ConnectionError):
            setup_manager.emit('test', {})

    def test_reconnection(self, setup_manager):
        """测试重连机制"""
        # 模拟断开连接
        setup_manager.disconnect()
        assert not setup_manager.connected

        # 测试自动重连
        assert setup_manager.reconnect()
        assert setup_manager.connected
        assert setup_manager.reconnect_attempts == 0

    def test_performance(self, setup_manager):
        """测试性能"""
        import time
        setup_manager.connect()

        # 测试发送延迟
        start_time = time.time()
        setup_manager.emit('test', {'data': 'test'})
        latency = time.time() - start_time
        assert latency < 0.001  # 发送延迟应小于1ms 

    def test_concurrent_connections(self):
        """测试并发连接"""
        connections = []
        for i in range(10):
            manager = SocketManager()
            assert manager.connect()
            connections.append(manager)
        
        # 验证所有连接都活跃
        assert all(m.connected for m in connections)

    def test_reconnection_with_data_recovery(self):
        """测试断线重连和数据恢复"""
        manager = self.setup_manager()
        test_data = []
        
        @manager.on('test_data')
        def handle_data(data):
            test_data.append(data)
        
        # 模拟断线重连
        manager.disconnect()
        time.sleep(0.1)
        assert manager.reconnect()
        
        # 验证数据恢复
        assert len(test_data) == len(self.original_data) 
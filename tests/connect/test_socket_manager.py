import pytest
import time
from connect.core.socket_manager import SocketManager, ConnectionConfig, ConnectionStatus
from connect.utils.errors import ConnectionError, ReconnectFailedError

class TestSocketManager:
    @pytest.fixture
    def setup_manager(self):
        """初始化测试环境"""
        config = ConnectionConfig(
            url='http://localhost:5000',
            reconnect_attempts=3,
            reconnect_delay=100
        )
        manager = SocketManager(config)
        yield manager
        manager.disconnect()

    def test_connection_lifecycle(self, setup_manager):
        """测试连接生命周期"""
        manager = setup_manager
        
        # 测试连接
        assert manager.connect()
        assert manager.connected
        
        # 测试状态
        status = manager.get_status()
        assert isinstance(status, ConnectionStatus)
        assert status.connected
        assert status.reconnect_count == 0
        
        # 测试断开
        manager.disconnect()
        assert not manager.connected

    def test_reconnection(self, setup_manager):
        """测试重连机制"""
        manager = setup_manager
        manager.connect()
        
        # 模拟断开
        manager.sio.disconnect()
        time.sleep(0.2)  # 等待重连
        
        # 验证重连
        status = manager.get_status()
        assert status.connected
        assert status.reconnect_count == 1

    def test_error_handling(self, setup_manager):
        """测试错误处理"""
        manager = setup_manager
        errors = []
        
        def error_handler(error_type: str, error: Exception):
            errors.append((error_type, error))
            
        manager.register_error_handler(error_handler)
        
        # 测试无效连接
        manager.config.url = "invalid_url"
        with pytest.raises(ConnectionError):
            manager.connect()
            
        assert len(errors) == 1
        assert isinstance(errors[0][1], ConnectionError) 
import pytest
import time
from connect.socket_manager import SocketManager, ConnectionConfig
from unittest.mock import Mock, patch

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
        manager.disconnect()  # 清理
        
    def test_connection_lifecycle(self, setup_manager):
        """测试连接生命周期"""
        manager = setup_manager
        
        # 测试连接
        assert manager.connect() == True
        assert manager.is_connected() == True
        
        # 测试断开
        manager.disconnect()
        assert manager.is_connected() == False
        
    def test_reconnection(self, setup_manager):
        """测试重连机制"""
        manager = setup_manager
        manager.connect()
        
        # 模拟断开
        manager.sio.disconnect()
        time.sleep(0.2)  # 等待重连
        
        assert manager.is_connected() == True
        
    def test_event_handling(self, setup_manager):
        """测试事件处理"""
        manager = setup_manager
        received_data = []
        
        @manager.on('test_event')
        def handle_event(data):
            received_data.append(data)
            
        manager.connect()
        manager.emit('test_event', {'test': 'data'})
        time.sleep(0.1)
        
        assert len(received_data) == 1
        assert received_data[0]['test'] == 'data' 
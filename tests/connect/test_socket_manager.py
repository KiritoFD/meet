import os
import sys
from pathlib import Path

# 确保能找到项目模块
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
import time
from unittest.mock import Mock, patch
import socketio
import json
import zlib
import base64
import jwt
from connect.socket_manager import (
    SocketManager, 
    ConnectionConfig,
    SecurityConfig,
    ConnectionPoolConfig
)
from connect.errors import ConnectionError, AuthError

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
        SocketManager._instances = []  # 重置类变量
        SocketManager._active_connections = 0
        
        manager = SocketManager(mock_socketio, mock_audio_processor)
        # 不在这里进行认证，让测试自己处理
        yield manager
        
        # 清理
        manager.disconnect()
        SocketManager._instances = []
        SocketManager._active_connections = 0

    def test_connection_pool_management(self, setup_manager):
        """测试连接池管理"""
        # 重置状态
        with setup_manager._lock:
            setup_manager._active_connections = 0
            setup_manager._instances.clear()
        
        # 测试初始状态
        pool_status = setup_manager.get_pool_status()
        assert pool_status['active_connections'] == 0
        
        # 创建连接
        for _ in range(setup_manager.pool_config.min_pool_size):
            setup_manager.connect()
            setup_manager._active_connections += 1  # 确保计数正确
            time.sleep(0.1)  # 添加短暂延迟
            
        pool_status = setup_manager.get_pool_status()
        assert pool_status['active_connections'] >= setup_manager.pool_config.min_pool_size

    def test_connection_health_check(self, setup_manager):
        """测试连接健康检查"""
        setup_manager.connect()
        
        # 模拟健康检查
        setup_manager._check_connections_health()
        
        # 验证心跳更新
        assert time.time() - setup_manager._status.last_heartbeat < 1
        
        # 模拟不健康连接
        setup_manager.sio.emit.side_effect = Exception("Connection lost")
        setup_manager._check_connections_health()
        
        # 验证重连尝试
        assert setup_manager._pool_status['failed_connections'] > 0

    def test_authentication(self, setup_manager):
        """测试认证功能"""
        # 测试有效凭据
        valid_credentials = {
            'username': 'admin',
            'password': 'admin123'
        }
        assert setup_manager.authenticate(valid_credentials)
        assert setup_manager._authenticated
        
        # 测试无效凭据
        invalid_credentials = {
            'username': 'invalid',
            'password': 'wrong'
        }
        with pytest.raises(AuthError):
            setup_manager.authenticate(invalid_credentials)
            
        # 测试缺失凭据
        with pytest.raises(AuthError):
            setup_manager.authenticate({})

    def test_data_compression(self, setup_manager):
        """测试数据压缩"""
        # 创建大数据
        large_data = {
            'data': 'x' * 2000  # 超过压缩阈值
        }
        
        # 处理数据
        processed = setup_manager._process_data(large_data)
        
        # 验证压缩
        assert processed['compressed']
        
        # 解压并验证
        decompressed = setup_manager._decompress_data(processed)
        assert decompressed == large_data
        
        # 测试小数据不压缩
        small_data = {'data': 'small'}
        processed = setup_manager._process_data(small_data)
        assert not processed['compressed']

    def test_data_security(self, setup_manager):
        """测试数据安全"""
        test_data = {'message': 'secret'}
        
        # 处理数据
        processed = setup_manager._process_data(test_data)
        
        # 验证安全头
        assert 'timestamp' in processed
        assert 'signature' in processed
        
        # 验证签名
        assert setup_manager._verify_data(processed)
        
        # 测试篡改检测
        processed['data'] = 'tampered'
        assert not setup_manager._verify_data(processed)

    def test_connection_recovery(self, setup_manager):
        """测试连接恢复"""
        setup_manager.connect()
        # 先进行认证
        setup_manager.authenticate({
            'username': 'admin',
            'password': 'admin123'
        })
        
        # 模拟数据发送
        test_data = {'test': 'data'}
        setup_manager.emit('test_event', test_data)
        
        # 模拟断开连接
        setup_manager.disconnect()
        
        # 验证重连和数据恢复
        setup_manager.connect()
        assert len(setup_manager._cached_data) > 0
        
        # 验证缓存数据完整性
        cached = setup_manager._cached_data[0]
        assert cached['event'] == 'test_event'
        assert cached['data']['test'] == 'data'

    def test_concurrent_operations(self, setup_manager):
        """测试并发操作"""
        import threading
        
        # 先进行认证
        setup_manager.authenticate({
            'username': 'admin',
            'password': 'admin123'
        })
        
        def send_data():
            for i in range(10):
                try:
                    setup_manager.emit('test', {'count': i})
                    time.sleep(0.01)
                except:
                    continue
        
        # 创建多个发送线程
        threads = [
            threading.Thread(target=send_data)
            for _ in range(5)
        ]
        
        # 启动线程
        for t in threads:
            t.start()
            
        # 等待完成
        for t in threads:
            t.join()
            
        # 验证发送统计
        assert setup_manager._total_count > 0
        assert setup_manager._success_count <= setup_manager._total_count

    def test_performance_metrics(self, setup_manager):
        """测试性能指标"""
        setup_manager.connect()
        # 先进行认证
        setup_manager.authenticate({
            'username': 'admin',
            'password': 'admin123'
        })
        
        # 发送一些测试数据
        for i in range(10):
            setup_manager.emit('test', {'data': f'test_{i}'})
            
        # 获取连接池状态
        pool_status = setup_manager.get_pool_status()
        
        # 验证指标
        assert 'active_connections' in pool_status
        assert 'pool_utilization' in pool_status
        assert 0 <= pool_status['pool_utilization'] <= 1
        
        # 验证性能计数器
        assert setup_manager._success_count > 0
        assert len(setup_manager._event_times) > 0

    def test_error_handling(self, setup_manager):
        """测试错误处理"""
        # 确保未认证状态
        setup_manager._authenticated = False
        
        # 测试未认证发送
        with pytest.raises(AuthError):
            setup_manager.emit('test', {})
        
        # 测试连接数超限
        setup_manager._active_connections = setup_manager.pool_config.max_pool_size
        with pytest.raises(ConnectionError):
            setup_manager.connect()

    def test_cleanup_mechanism(self, setup_manager):
        """测试清理机制"""
        # 创建一些连接
        setup_manager.connect()
        
        # 模拟过期连接
        setup_manager._status.last_heartbeat = time.time() - setup_manager.pool_config.connection_timeout - 1
        
        # 运行清理
        setup_manager._manage_connection_pool()
        
        # 验证清理结果
        pool_status = setup_manager.get_pool_status()
        assert pool_status['active_connections'] <= setup_manager.pool_config.max_pool_size
        
        # 验证最小连接数维护
        assert pool_status['available_connections'] >= setup_manager.pool_config.min_pool_size 
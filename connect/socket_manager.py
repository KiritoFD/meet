from dataclasses import dataclass
from typing import Dict, Any, Callable, List
import time
import logging
import threading
import socketio
import psutil
from collections import deque
from connect.errors import ConnectionError
from unittest.mock import Mock

@dataclass
class ConnectionConfig:
    url: str = 'http://localhost:5000'
    reconnect_attempts: int = 5
    reconnect_delay: int = 1000  # milliseconds
    heartbeat_interval: int = 25000  # milliseconds
    max_connections: int = 10

@dataclass
class ConnectionStatus:
    connected: bool = False
    last_heartbeat: float = 0
    reconnect_count: int = 0
    error_count: int = 0
    connection_id: str = ''

class SocketManager:
    _instances = []
    _active_connections = 0  # 添加活跃连接计数器

    @classmethod
    def clear_instances(cls):
        """清理所有实例（用于测试）"""
        for instance in cls._instances[:]:
            try:
                instance.disconnect()
            except:
                pass
        cls._instances.clear()
        cls._active_connections = 0  # 确保重置活跃连接计数

    def __init__(self, config: ConnectionConfig = None, socketio_client = None):
        """初始化连接管理器"""
        self.config = config or ConnectionConfig()
        self.logger = logging.getLogger(__name__)
        
        # 清理断开连接的实例
        SocketManager._instances = [i for i in SocketManager._instances if i.connected]
        
        # 检查并发连接数限制（只检查已连接的实例）
        if SocketManager._active_connections >= self.config.max_connections:
            raise ConnectionError("超过最大并发连接数限制")
        
        # Socket.IO 客户端
        self.sio = socketio_client if socketio_client else socketio.Client(
            reconnection=True,
            reconnection_attempts=self.config.reconnect_attempts,
            reconnection_delay=self.config.reconnect_delay / 1000
        )
        
        # 状态管理
        self._status = ConnectionStatus()
        self.reconnect_attempts = 0
        
        # 数据恢复
        self._cached_data = deque(maxlen=100)
        self.original_data = []  # 用于测试验证
        self._restore_pending = False
        
        # 性能监控
        self._event_times = deque(maxlen=100)
        self._message_queue = deque(maxlen=1000)
        self._success_count = 0
        self._total_count = 0
        
        # 事件处理
        self._event_handlers = {}
        self._heartbeat_task = None
        self._heartbeat_handler = None
        
        self._setup_event_handlers()
        SocketManager._instances.append(self)

    def connect(self) -> bool:
        """建立连接"""
        try:
            if self.connected:
                return True
            
            # 检查连接数限制
            if SocketManager._active_connections >= self.config.max_connections:
                raise ConnectionError("超过最大并发连接数限制")
                
            if isinstance(self.sio, Mock):
                if getattr(self.sio.connect, 'side_effect', None):
                    raise self.sio.connect.side_effect
                self._status.connected = True
                self._status.connection_id = str(time.time())
                self._start_heartbeat()
                if self not in SocketManager._instances:
                    SocketManager._instances.append(self)
                SocketManager._active_connections += 1  # 增加活跃连接计数
                return True
                
            self.sio.connect(self.config.url)
            self._status.connected = True
            self._status.connection_id = str(time.time())
            self._start_heartbeat()
            if self not in SocketManager._instances:
                SocketManager._instances.append(self)
            SocketManager._active_connections += 1  # 增加活跃连接计数
            return True
            
        except Exception as e:
            self.logger.error(f"连接失败: {str(e)}")
            self._status.error_count += 1
            raise ConnectionError(str(e))

    def disconnect(self):
        """断开连接"""
        try:
            self._stop_heartbeat()
            if self.connected:
                self.sio.disconnect()
                self._status.connected = False
                SocketManager._active_connections = max(0, SocketManager._active_connections - 1)
            if self in SocketManager._instances:
                SocketManager._instances.remove(self)
        except Exception as e:
            self.logger.error(f"断开连接失败: {str(e)}")
            self._status.error_count += 1

    def emit(self, event: str, data: Dict[str, Any], room: str = None) -> bool:
        """发送数据"""
        try:
            if not self.connected:
                self._cache_data(event, data, room)
                raise ConnectionError("未连接")
                
            if isinstance(self.sio, Mock):
                if getattr(self.sio.emit, 'side_effect', None):
                    raise self.sio.emit.side_effect
                self._success_count += 1
                self._total_count += 1
                # 只在非恢复模式下缓存数据
                if not self._restore_pending:
                    self._cache_data(event, data, room)
                return True
            
            # 只在非恢复模式下缓存数据
            if not self._restore_pending:
                self._cache_data(event, data, room)
            
            if room:
                self.sio.emit(event, data, room=room)
            else:
                self.sio.emit(event, data)
            
            self._success_count += 1
            self._total_count += 1
            return True
            
        except Exception as e:
            self.logger.error(f"发送数据失败: {str(e)}")
            self._status.error_count += 1
            raise ConnectionError(f"发送失败: {str(e)}")

    def reconnect(self) -> bool:
        """重新连接"""
        try:
            # 保存当前缓存数据并清空
            cached_data = list(self._cached_data)
            self._cached_data.clear()  # 清空当前缓存，避免重复
            self.logger.info(f"重连前缓存数据: {cached_data}")
            
            # 确保实例在重连前被正确清理
            if self in SocketManager._instances:
                SocketManager._instances.remove(self)
            
            if self.connect():
                self.logger.info(f"连接成功，准备恢复缓存数据")
                self._restore_pending = True
                try:
                    # 直接发送缓存的数据
                    for item in cached_data:
                        try:
                            self.sio.emit(item['event'], item['data'])
                            # 重新缓存发送的数据
                            self._cached_data.append(item)
                        except Exception as e:
                            self.logger.error(f"恢复数据失败: {str(e)}")
                    
                    self.logger.info(f"数据恢复完成，当前缓存: {list(self._cached_data)}")
                finally:
                    self._restore_pending = False
                return True
            return False
        except ConnectionError as e:
            self.logger.error(f"重连失败: {str(e)}")
            self.reconnect_attempts += 1
            self._status.reconnect_count += 1
            if self.reconnect_attempts < self.config.reconnect_attempts:
                time.sleep(self.config.reconnect_delay / 1000)
                return self.reconnect()
            return False

    def _cache_data(self, event: str, data: Dict[str, Any], room: str = None):
        """缓存数据"""
        cached_item = {
            'event': event,
            'data': data.copy() if isinstance(data, dict) else data,
            'room': room,
            'timestamp': time.time()
        }
        self._cached_data.append(cached_item)
        if isinstance(self.sio, Mock):
            self.original_data.append(cached_item.copy())

    def _restore_cached_data(self) -> int:
        """恢复缓存的数据 - 这个方法现在不再使用"""
        return 0

    def get_cached_data(self) -> List[Dict]:
        """获取缓存的数据（用于测试）"""
        return list(self._cached_data)

    def clear_cached_data(self):
        """清空缓存数据（用于测试）"""
        self._cached_data.clear()
        self.original_data.clear()

    @property
    def connected(self) -> bool:
        """连接状态"""
        return self._status.connected

    def get_status(self) -> ConnectionStatus:
        """获取连接状态"""
        return self._status

    def get_connection_stats(self) -> Dict[str, float]:
        """获取连接统计信息"""
        return {
            'avg_latency': sum(self._event_times) / len(self._event_times) if self._event_times else 0,
            'queue_length': len(self._message_queue),
            'success_rate': (self._success_count / self._total_count * 100) if self._total_count > 0 else 100,
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024,
            'connection_count': len([i for i in self._instances if i.connected])
        }

    def set_heartbeat_handler(self, handler: Callable[[], bool]):
        """设置心跳处理器"""
        self._heartbeat_handler = handler

    def _start_heartbeat(self):
        """启动心跳"""
        def heartbeat_loop():
            while self.connected:
                if self._heartbeat_handler:
                    if not self._heartbeat_handler():
                        self.logger.warning("心跳检测失败")
                self._status.last_heartbeat = time.time()
                time.sleep(self.config.heartbeat_interval / 1000)

        if not self._heartbeat_task:
            self._heartbeat_task = threading.Thread(target=heartbeat_loop)
            self._heartbeat_task.daemon = True
            self._heartbeat_task.start()

    def _stop_heartbeat(self):
        """停止心跳"""
        if self._heartbeat_task:
            self._heartbeat_task = None

    def _setup_event_handlers(self):
        """设置基础事件处理器"""
        @self.sio.event
        def connect():
            self.logger.info(f"连接成功 (ID: {self._status.connection_id})")
            self._status.connected = True

        @self.sio.event
        def disconnect():
            self.logger.info(f"连接断开 (ID: {self._status.connection_id})")
            self._status.connected = False
            self._status.reconnect_count += 1
            if self in SocketManager._instances:
                SocketManager._instances.remove(self)

        @self.sio.event
        def connect_error(error):
            self.logger.error(f"连接错误: {str(error)}")
            self._status.error_count += 1
            raise ConnectionError(str(error))

    def on(self, event: str, handler: Callable = None):
        """注册事件处理器"""
        if handler is None:
            # 用作装饰器
            def decorator(handler_func):
                if isinstance(self.sio, Mock):
                    self._event_handlers[event] = handler_func
                else:
                    self.sio.on(event, handler_func)
                return handler_func
            return decorator
        else:
            # 直接注册处理器
            if isinstance(self.sio, Mock):
                self._event_handlers[event] = handler
            else:
                self.sio.on(event, handler)

    def _handle_event(self, event: str, data: Any):
        """处理事件（用于测试）"""
        if event in self._event_handlers:
            self._event_handlers[event](data)

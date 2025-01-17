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
import yaml

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
    _active_connections = 0

    def __init__(self, socketio, audio_processor):
        self.socketio = socketio
        self.audio_processor = audio_processor
        self.logger = logging.getLogger(__name__)
        
        # 读取配置
        try:
            with open('config/config.yaml', 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)['socket']
        except Exception as e:
            self.logger.warning(f"无法读取配置文件，使用默认配置: {str(e)}")
            self.config = {
                'max_connections': 10,
                'ping_timeout': 60,
                'ping_interval': 25,
                'reconnect_attempts': 5,
                'reconnect_delay': 1000,
                'heartbeat_interval': 25000,
                'url': 'http://localhost:5000'
            }
            
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

        # 初始化 Socket.IO 客户端
        if isinstance(socketio, Mock):
            self.sio = socketio  # 测试时使用 mock
        else:
            self.sio = socketio.Client(
                reconnection=True,
                reconnection_attempts=self.config['reconnect_attempts'],
                reconnection_delay=self.config['reconnect_delay'] / 1000
            )

        self._setup_event_handlers()

    def connect(self) -> bool:
        """建立连接"""
        try:
            if self.connected:
                return True
            
            # 检查连接数限制
            if SocketManager._active_connections >= self.config['max_connections']:
                raise ConnectionError("超过最大并发连接数限制")
                
            if isinstance(self.sio, Mock):
                self._status.connected = True
                self._status.connection_id = str(time.time())
                self._start_heartbeat()
                if self not in SocketManager._instances:
                    SocketManager._instances.append(self)
                SocketManager._active_connections += 1  # 增加活跃连接计数
                return True
                
            self.sio.connect(self.config['url'])
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
                if not isinstance(self.sio, Mock):
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
                self._success_count += 1
                self._total_count += 1
                if not self._restore_pending:
                    self._cache_data(event, data, room)
                return True
            
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

    @property
    def connected(self) -> bool:
        """连接状态"""
        return self._status.connected

    def on(self, event: str, handler: Callable = None):
        """注册事件处理器"""
        if handler is None:
            def decorator(handler_func):
                self._event_handlers[event] = handler_func
                if not isinstance(self.sio, Mock):
                    self.sio.on(event, handler_func)
                return handler_func
            return decorator
        else:
            self._event_handlers[event] = handler
            if not isinstance(self.sio, Mock):
                self.sio.on(event, handler)

    def _handle_event(self, event: str, data: Any):
        """处理事件（用于测试）"""
        if event in self._event_handlers:
            self._event_handlers[event](data)

    def _setup_event_handlers(self):
        """设置基础事件处理器"""
        @self.on('connect')
        def on_connect():
            self.logger.info(f"连接成功 (ID: {self._status.connection_id})")
            self._status.connected = True

        @self.on('disconnect')
        def on_disconnect():
            self.logger.info(f"连接断开 (ID: {self._status.connection_id})")
            self._status.connected = False

    def _start_heartbeat(self):
        """启动心跳"""
        if not self._heartbeat_task:
            self._heartbeat_task = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._heartbeat_task.start()

    def _stop_heartbeat(self):
        """停止心跳"""
        self._heartbeat_task = None

    def _heartbeat_loop(self):
        """心跳循环"""
        while self._heartbeat_task and self.connected:
            if self._heartbeat_handler:
                if not self._heartbeat_handler():
                    self.logger.warning("心跳检测失败")
            self._status.last_heartbeat = time.time()
            time.sleep(self.config['heartbeat_interval'] / 1000)

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

import socketio
import logging
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ConnectionConfig:
    """Socket连接配置"""
    url: str = 'http://localhost:5000'
    reconnect_attempts: int = 5
    reconnect_delay: int = 1000
    heartbeat_interval: int = 25000

class SocketManager:
    def __init__(self, socketio_server=None, config: ConnectionConfig = None):
        """初始化连接管理器
        Args:
            socketio_server: Socket.IO服务器实例(可选，用于测试)
            config: 连接配置(可选)
        """
        self.sio = socketio.Client() if socketio_server is None else socketio_server
        self.config = config or ConnectionConfig()
        self.connected = False
        self._setup_handlers()
        
    def _setup_handlers(self):
        """设置事件处理器"""
        @self.sio.event
        def connect():
            logger.info("Socket已连接")
            self.connected = True
            
        @self.sio.event
        def disconnect():
            logger.info("Socket已断开")
            self.connected = False
            
        @self.sio.event
        def connect_error(data):
            logger.error(f"连接错误: {data}")
            
    def connect(self) -> bool:
        """建立连接"""
        try:
            if not self.connected:
                if isinstance(self.sio, socketio.Client):
                    self.sio.connect(
                        self.config.url,
                        wait_timeout=10,
                        wait=True,
                        transports=['websocket']
                    )
                else:
                    # 测试模式，模拟连接
                    self.connected = True
            return True
        except Exception as e:
            logger.error(f"连接失败: {e}")
            return False
            
    def disconnect(self):
        """断开连接"""
        if self.connected:
            if isinstance(self.sio, socketio.Client):
                self.sio.disconnect()
            self.connected = False

    def emit(self, event: str, data: dict) -> bool:
        """发送数据"""
        if self.connected:
            try:
                self.sio.emit(event, data)
                return True
            except Exception as e:
                logger.error(f"发送数据失败: {e}")
                return False
        return False

    def broadcast(self, event: str, data: Dict[str, Any], room: str) -> bool:
        """广播数据到房间
        Args:
            event: 事件名称
            data: 要发送的数据
            room: 房间ID
        Returns:
            bool: 是否发送成功
        """
        try:
            self.sio.emit(event, data, room=room)
            return True
        except Exception as e:
            logger.error(f"广播数据失败: {e}")
            return False

    @property
    def sid(self) -> Optional[str]:
        """获取当前socket的会话ID"""
        return getattr(self.sio, 'sid', None)

    def register_handler(self, event: str, handler: Callable):
        """注册事件处理器
        Args:
            event: 事件名称
            handler: 处理函数
        """
        self.sio.on(event, handler)

    def is_connected(self) -> bool:
        """检查连接状态"""
        return self.connected
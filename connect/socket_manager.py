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
    def __init__(self, socketio, audio_processor=None, config: ConnectionConfig = None):
        """初始化连接管理器
        Args:
            socketio: Socket.IO客户端实例
            audio_processor: 音频处理器(可选)
            config: 连接配置(可选)
        """
        self.sio = socketio
        self.audio_processor = audio_processor
        self.config = config or ConnectionConfig()
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
                self.sio.connect(
                    self.config.url,
                    reconnection=self.config.reconnection,
                    reconnection_attempts=self.config.reconnection_attempts,
                    reconnection_delay=self.config.reconnection_delay
                )
            return True
        except Exception as e:
            logger.error(f"连接失败: {e}")
            return False
            
    def disconnect(self):
        """断开连接"""
        if self.connected:
            self.sio.disconnect()
            
    def emit(self, event: str, data: dict):
        """发送数据"""
        if self.connected:
            try:
                self.sio.emit(event, data)
                return True
            except Exception as e:
                logger.error(f"发送数据失败: {e}")
                return False
        return False 
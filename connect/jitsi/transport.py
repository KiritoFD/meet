from typing import Dict, List, Optional, Callable
from collections import deque
import asyncio
import time
from lib.jitsi import (
    JitsiMeet,
    JitsiConnection,
    JitsiConference,
    JitsiDataChannel
)

class TransportError(Exception):
    pass

class JitsiTransport:
    def __init__(self, config: Dict):
        """初始化传输层
        
        Args:
            config: 配置字典，包含:
                - buffer_size: 缓冲区大小
                - retry_limit: 重试次数
                - timeout: 超时时间(秒)
                - batch_size: 批量发送大小
        """
        self.config = config
        self._data_handlers = []
        self._buffer = deque(maxlen=config['buffer_size'])
        self._stats = {
            'sent': 0,
            'received': 0,
            'errors': 0,
            'latency': deque(maxlen=100)
        }
        
        # Jitsi 组件
        self._jitsi = None
        self._connection = None
        self._conference = None
        self._data_channel = None
        
    def on_data(self, handler: Callable):
        """注册数据处理回调"""
        self._data_handlers.append(handler)
        return handler
        
    async def connect(self, room_id: str) -> bool:
        """连接到 Jitsi 会议室"""
        try:
            self._jitsi = JitsiMeet(
                host=self.config['jitsi_host'],
                port=self.config['jitsi_port']
            )
            
            self._connection = await self._jitsi.connect()
            self._conference = await self._connection.join_conference(room_id)
            
            channel_options = self.config['conference']['data_channel_options']
            self._data_channel = await self._conference.create_data_channel(
                "pose_data", channel_options
            )
            
            # 设置事件处理器
            self._data_channel.on_message = self._handle_message
            self._data_channel.on_open = self._handle_open
            self._data_channel.on_close = self._handle_close
            
            return True
            
        except Exception as e:
            raise TransportError(f"Failed to connect: {str(e)}")
            
    async def send(self, data: bytes) -> bool:
        """发送数据"""
        if not self._data_channel or not self._data_channel.ready():
            await self._ensure_connection()
            
        try:
            start_time = time.time()
            await self._data_channel.send(data)
            self._stats['latency'].append(time.time() - start_time)
            self._stats['sent'] += 1
            return True
            
        except Exception as e:
            self._stats['errors'] += 1
            raise TransportError(f"Send failed: {str(e)}")
            
    async def batch_send(self, data_list: List[bytes]) -> List[bool]:
        """批量发送数据"""
        if len(data_list) > self.config['batch_size']:
            chunks = [data_list[i:i + self.config['batch_size']] 
                     for i in range(0, len(data_list), self.config['batch_size'])]
            results = []
            for chunk in chunks:
                chunk_results = await asyncio.gather(
                    *[self.send(data) for data in chunk],
                    return_exceptions=True
                )
                results.extend(chunk_results)
            return results
        else:
            return await asyncio.gather(
                *[self.send(data) for data in data_list],
                return_exceptions=True
            )
            
    async def _ensure_connection(self):
        """确保连接可用"""
        if not self._data_channel or not self._data_channel.ready():
            if self.config['conference']['room_id']:
                await self.connect(self.config['conference']['room_id'])
            else:
                raise TransportError("No room_id available for reconnection")
                
    async def _handle_message(self, data: bytes):
        """处理接收到的数据"""
        self._stats['received'] += 1
        for handler in self._data_handlers:
            try:
                await handler(data)
            except Exception as e:
                self._stats['errors'] += 1
                
    def _handle_open(self):
        """处理通道打开事件"""
        pass
        
    def _handle_close(self):
        """处理通道关闭事件"""
        pass
        
    async def close(self):
        """关闭所有连接"""
        if self._data_channel:
            await self._data_channel.close()
        if self._conference:
            await self._conference.leave()
        if self._connection:
            await self._connection.disconnect()
            
        self._data_channel = None
        self._conference = None
        self._connection = None
        self._buffer.clear()
        
    def get_stats(self) -> Dict:
        """获取传输统计信息"""
        return {
            'sent': self._stats['sent'],
            'received': self._stats['received'],
            'errors': self._stats['errors'],
            'avg_latency': sum(self._stats['latency']) / len(self._stats['latency'])
                if self._stats['latency'] else 0,
            'connection_state': self._data_channel.get_state() if self._data_channel else 'closed'
        } 
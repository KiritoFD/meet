from typing import Dict, Optional, Any
import asyncio
from lib.jitsi import JitsiMeet
from lib import (
    JitsiConnection as BaseConnection,
    JitsiConference as BaseConference
)

class JitsiClient:
    def __init__(self, config: Dict):
        self.config = config
        self._jitsi = JitsiMeet(
            host=config['jitsi_host'],
            port=config['jitsi_port'],
            ice_servers=config['ice_servers']
        )
        self._connection = None
        self._conference = None
        
    async def connect(self) -> bool:
        """建立Jitsi连接"""
        try:
            self._connection = await self._jitsi.connect(
                token=self.config.get('token')
            )
            return True
        except Exception as e:
            raise ConnectionError(f"Jitsi connection failed: {e}")
            
    async def join_room(self, room_id: str) -> Any:
        """加入会议室"""
        if not self._connection:
            await self.connect()
            
        try:
            self._conference = await self._connection.join_conference(room_id)
            return self._conference
        except Exception as e:
            raise ConnectionError(f"Failed to join room: {e}")
            
    async def create_data_channel(self, label: str, options: Dict) -> Any:
        """创建数据通道"""
        if not self._conference:
            raise RuntimeError("Not in conference")
            
        return await self._conference.create_data_channel(label, options)
        
    async def close(self):
        """关闭所有连接"""
        if self._conference:
            await self._conference.leave()
        if self._connection:
            await self._connection.disconnect()

class JitsiMeetingController:
    def __init__(self):
        self._hand_raise_queue = []  # 举手队列
        self._current_speaker = None  # 当前发言人
        
    async def request_speak(self, user_id: str):
        """申请发言"""
        if user_id not in self._hand_raise_queue:
            self._hand_raise_queue.append(user_id)
            
    async def grant_speak(self, user_id: str):
        """授予发言权"""
        if user_id in self._hand_raise_queue:
            self._hand_raise_queue.remove(user_id)
            self._current_speaker = user_id
            
    async def revoke_speak(self):
        """收回发言权"""
        self._current_speaker = None 
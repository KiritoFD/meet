from typing import Dict, Set, Optional
from dataclasses import dataclass
import time
import asyncio
from .transport import JitsiTransport

@dataclass
class JitsiRoom:
    room_id: str
    host_id: str
    participants: Set[str]
    created_at: float = time.time()
    last_active: float = time.time()

class JitsiMeetingManager:
    def __init__(self, config: Dict):
        self.config = config
        self.rooms: Dict[str, JitsiRoom] = {}
        self.user_rooms: Dict[str, str] = {}  # user_id -> room_id
        self._cleanup_task = None
        
    async def start(self):
        """启动会议管理器"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
    async def create_meeting(self, room_id: str, host_id: str) -> str:
        """创建新会议"""
        if room_id in self.rooms:
            raise ValueError(f"Room {room_id} already exists")
            
        room = JitsiRoom(
            room_id=room_id,
            host_id=host_id,
            participants={host_id}
        )
        self.rooms[room_id] = room
        self.user_rooms[host_id] = room_id
        return room_id
        
    async def join_meeting(self, room_id: str, user_id: str) -> bool:
        """加入会议"""
        if room_id not in self.rooms:
            raise ValueError(f"Room {room_id} not found")
            
        room = self.rooms[room_id]
        if len(room.participants) >= self.config['conference']['max_participants']:
            return False
            
        room.participants.add(user_id)
        self.user_rooms[user_id] = room_id
        room.last_active = time.time()
        return True
        
    async def leave_meeting(self, room_id: str, user_id: str):
        """离开会议"""
        if room_id in self.rooms:
            room = self.rooms[room_id]
            room.participants.discard(user_id)
            self.user_rooms.pop(user_id, None)
            
            # 如果房间空了，清理它
            if not room.participants:
                await self._cleanup_room(room_id)
                
    async def _cleanup_loop(self):
        """定期清理空闲房间"""
        while True:
            try:
                await self._cleanup_idle_rooms()
            except Exception as e:
                print(f"Cleanup error: {e}")
            await asyncio.sleep(self.config['cleanup_interval'])
            
    async def _cleanup_idle_rooms(self):
        """清理空闲房间"""
        now = time.time()
        idle_timeout = self.config['idle_timeout']
        
        for room_id, room in list(self.rooms.items()):
            if now - room.last_active > idle_timeout:
                await self._cleanup_room(room_id)
                
    async def _cleanup_room(self, room_id: str):
        """清理单个房间"""
        if room_id in self.rooms:
            room = self.rooms.pop(room_id)
            for user_id in room.participants:
                self.user_rooms.pop(user_id, None) 
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
import time

class SocketManager:
    def __init__(self, socketio, audio_processor=None, config=None):
        self.socketio = socketio
        self.audio_processor = audio_processor
        self.config = config or ConnectionConfig()
        self.connected = True  # Flask-SocketIO 会自动管理连接
        self.handlers = {}
        self.logger = logging.getLogger(__name__)
        self.room_manager = RoomManager()

    async def emit(self, event: str, data: Any, room: str = None) -> bool:
        """异步发送数据"""
        try:
            if room:
                self.socketio.emit(event, data, room=room)
            else:
                self.socketio.emit(event, data)
            return True
        except Exception as e:
            self.logger.error(f"Error sending data: {str(e)}")
            return False

    def on(self, event: str, handler: Callable):
        """注册事件处理器"""
        if event not in self.handlers:
            self.handlers[event] = []
        self.handlers[event].append(handler)
        self.socketio.on(event)(handler)

    def join_room(self, room: str, sid: str = None):
        """加入房间"""
        self.room_manager.join_room(room, sid)

    def leave_room(self, room: str, sid: str = None):
        """离开房间"""
        self.room_manager.leave_room(room, sid)

    def is_connected(self) -> bool:
        """检查连接状态"""
        return self.connected

class RoomManager:
    def __init__(self):
        self.rooms = {}
        self.user_rooms = {}

    def join_room(self, room: str, sid: str = None):
        if room not in self.rooms:
            self.rooms[room] = set()
        if sid:
            self.rooms[room].add(sid)
            if sid not in self.user_rooms:
                self.user_rooms[sid] = set()
            self.user_rooms[sid].add(room)

    def leave_room(self, room: str, sid: str = None):
        if room in self.rooms and sid in self.rooms[room]:
            self.rooms[room].remove(sid)
            if sid in self.user_rooms:
                self.user_rooms[sid].remove(room)

    def get_room_members(self, room: str) -> set:
        return self.rooms.get(room, set())

    def get_user_rooms(self, sid: str) -> set:
        return self.user_rooms.get(sid, set())

@dataclass
class ConnectionConfig:
    url: str = "ws://localhost:8765"
    reconnect_attempts: int = 5
    reconnect_delay: int = 1
    heartbeat_interval: int = 30 
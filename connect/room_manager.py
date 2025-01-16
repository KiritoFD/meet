from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import time
import threading
from .socket_manager import SocketManager

@dataclass
class RoomConfig:
    max_clients: int = 10
    timeout: int = 30  # 房间超时时间(秒)
    auto_cleanup: bool = True

@dataclass
class RoomStatus:
    room_id: str
    member_count: int
    created_time: float
    last_active: float
    is_locked: bool

@dataclass
class RoomMember:
    id: str
    join_time: float
    last_active: float
    role: str = 'member'  # 'host' or 'member'

class RoomManager:
    def __init__(self, socket: SocketManager, config: RoomConfig = None):
        self.socket = socket
        self.config = config or RoomConfig()
        self._rooms: Dict[str, Dict] = {}  # 房间字典
        self._members: Dict[str, Dict[str, RoomMember]] = {}  # 房间成员字典
        self._user_room: Dict[str, str] = {}  # 用户当前所在房间
        self._lock = threading.Lock()
        
        if self.config.auto_cleanup:
            self._start_cleanup_timer()

    def create_room(self, room_id: str) -> bool:
        with self._lock:
            if room_id in self._rooms:
                return False
            
            self._rooms[room_id] = {
                'created_time': time.time(),
                'last_active': time.time(),
                'is_locked': False
            }
            self._members[room_id] = {}
            return True

    def join_room(self, room_id: str, user_id: str) -> bool:
        with self._lock:
            if room_id not in self._rooms:
                return False
                
            if len(self._members[room_id]) >= self.config.max_clients:
                return False
                
            # 如果用户在其他房间，先离开
            if user_id in self._user_room:
                self.leave_room(self._user_room[user_id], user_id)
            
            current_time = time.time()
            self._members[room_id][user_id] = RoomMember(
                id=user_id,
                join_time=current_time,
                last_active=current_time,
                role='host' if len(self._members[room_id]) == 0 else 'member'
            )
            self._user_room[user_id] = room_id
            self._rooms[room_id]['last_active'] = current_time
            return True

    def leave_room(self, room_id: str, user_id: str):
        """离开房间"""
        with self._lock:
            if room_id not in self._rooms or user_id not in self._members.get(room_id, {}):
                return
            
            # 移除成员
            del self._members[room_id][user_id]
            if user_id in self._user_room:
                del self._user_room[user_id]
            
            # 如果房间空了，删除房间
            if not self._members[room_id]:
                del self._rooms[room_id]
                del self._members[room_id]

    def broadcast(self, event: str, data: Dict[str, Any], room_id: str) -> bool:
        if room_id not in self._rooms:
            return False
        
        self._rooms[room_id]['last_active'] = time.time()
        return self.socket.broadcast(event, data, room=room_id)

    def get_room_info(self, room_id: str) -> Optional[RoomStatus]:
        if room_id not in self._rooms:
            return None
            
        room = self._rooms[room_id]
        return RoomStatus(
            room_id=room_id,
            member_count=len(self._members[room_id]),
            created_time=room['created_time'],
            last_active=room['last_active'],
            is_locked=room['is_locked']
        )

    def list_rooms(self) -> List[RoomStatus]:
        return [self.get_room_info(room_id) for room_id in self._rooms.keys()]

    def clean_inactive_rooms(self):
        """清理不活跃房间"""
        current_time = time.time()
        
        # 先获取需要清理的房间列表
        with self._lock:
            inactive_rooms = [
                room_id for room_id, room in self._rooms.items()
                if current_time - room['last_active'] > self.config.timeout
            ]
        
        # 在锁外处理清理操作
        for room_id in inactive_rooms:
            with self._lock:
                if room_id not in self._rooms:
                    continue
                # 获取房间所有成员
                members = list(self._members[room_id].keys())
            
            # 在锁外逐个处理成员离开
            for member_id in members:
                self.leave_room(room_id, member_id)

    @property
    def current_room(self) -> Optional[str]:
        return self._user_room.get(self.socket.sid)

    def get_members(self, room_id: str) -> List[RoomMember]:
        if room_id not in self._members:
            return []
        return list(self._members[room_id].values())

    def update_member_status(self, room_id: str, member_id: str):
        if room_id in self._members and member_id in self._members[room_id]:
            self._members[room_id][member_id].last_active = time.time()
            self._rooms[room_id]['last_active'] = time.time()

    def set_member_role(self, room_id: str, member_id: str, role: str):
        if role not in ['host', 'member']:
            raise ValueError("Invalid role")
            
        if room_id in self._members and member_id in self._members[room_id]:
            self._members[room_id][member_id].role = role

    def kick_member(self, room_id: str, member_id: str):
        if room_id in self._rooms and member_id in self._members[room_id]:
            self.leave_room(room_id, member_id)
            # 可以在这里发送被踢出的通知

    def _start_cleanup_timer(self):
        """启动定时清理任务"""
        def cleanup():
            while True:
                self.clean_inactive_rooms()
                time.sleep(self.config.timeout)
                
        thread = threading.Thread(target=cleanup, daemon=True)
        thread.start() 
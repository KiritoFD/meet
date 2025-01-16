import logging
from typing import Dict, Set

logger = logging.getLogger(__name__)

class RoomManager:
    def __init__(self):
        self.rooms: Dict[str, Set[str]] = {}  # room_id -> set of user_ids
        
    def create_room(self, room_id: str) -> bool:
        """创建房间"""
        if room_id in self.rooms:
            return False
        self.rooms[room_id] = set()
        logger.info(f"创建房间: {room_id}")
        return True
        
    def join_room(self, room_id: str, user_id: str) -> bool:
        """用户加入房间"""
        if room_id not in self.rooms:
            self.create_room(room_id)
        self.rooms[room_id].add(user_id)
        logger.info(f"用户 {user_id} 加入房间 {room_id}")
        return True
        
    def leave_room(self, room_id: str, user_id: str) -> bool:
        """用户离开房间"""
        if room_id not in self.rooms:
            return False
        self.rooms[room_id].discard(user_id)
        if not self.rooms[room_id]:  # 如果房间空了就删除
            del self.rooms[room_id]
        logger.info(f"用户 {user_id} 离开房间 {room_id}")
        return True
        
    def get_room_users(self, room_id: str) -> Set[str]:
        """获取房间内的所有用户"""
        return self.rooms.get(room_id, set())
        
    def room_exists(self, room_id: str) -> bool:
        """检查房间是否存在"""
        return room_id in self.rooms 
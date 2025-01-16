from typing import Dict, Set
import logging

logger = logging.getLogger(__name__)

class RoomManager:
    def __init__(self):
        self.rooms: Dict[str, Set[str]] = {}  # room_id -> set of sid
        self.user_rooms: Dict[str, str] = {}  # sid -> room_id
        
    def join_room(self, room_id: str, sid: str) -> bool:
        """用户加入房间"""
        try:
            # 确保用户离开之前的房间
            self.leave_current_room(sid)
            
            # 创建房间(如果不存在)
            if room_id not in self.rooms:
                self.rooms[room_id] = set()
            
            # 加入新房间
            self.rooms[room_id].add(sid)
            self.user_rooms[sid] = room_id
            
            logger.info(f"用户 {sid} 加入房间 {room_id}")
            return True
            
        except Exception as e:
            logger.error(f"加入房间失败: {str(e)}")
            return False
    
    def leave_current_room(self, sid: str) -> None:
        """用户离开当前房间"""
        if sid in self.user_rooms:
            room_id = self.user_rooms[sid]
            if room_id in self.rooms:
                self.rooms[room_id].remove(sid)
                # 如果房间空了就删除
                if not self.rooms[room_id]:
                    del self.rooms[room_id]
            del self.user_rooms[sid]
            logger.info(f"用户 {sid} 离开房间 {room_id}")
    
    def get_room_members(self, room_id: str) -> Set[str]:
        """获取房间成员"""
        return self.rooms.get(room_id, set())
    
    def get_user_room(self, sid: str) -> str:
        """获取用户当前所在房间"""
        return self.user_rooms.get(sid)
    
    def room_exists(self, room_id: str) -> bool:
        """检查房间是否存在"""
        return room_id in self.rooms 
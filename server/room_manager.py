from typing import Dict, List, Optional, Set
import time
import logging
from dataclasses import dataclass
import json

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class User:
    id: str
    name: str
    room_id: Optional[str] = None
    connected: bool = False
    last_active: float = None
    
    def __post_init__(self):
        if self.last_active is None:
            self.last_active = time.time()

@dataclass
class Room:
    id: str
    name: str
    owner_id: str
    max_users: int = 2
    created_at: float = None
    users: Set[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.users is None:
            self.users = set()
            
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "owner_id": self.owner_id,
            "max_users": self.max_users,
            "created_at": self.created_at,
            "users": list(self.users)
        }

class RoomManager:
    def __init__(self):
        """初始化房间管理器"""
        self.rooms: Dict[str, Room] = {}
        self.users: Dict[str, User] = {}
        self.inactive_timeout = 300  # 5分钟不活跃视为断开
        self.cleanup_interval = 60  # 每60秒清理一次
        self.last_cleanup = time.time()
        
    def create_room(self, room_id: str, room_name: str, owner_id: str) -> Room:
        """创建新房间"""
        try:
            if room_id in self.rooms:
                raise ValueError(f"房间ID {room_id} 已存在")
                
            room = Room(
                id=room_id,
                name=room_name,
                owner_id=owner_id
            )
            self.rooms[room_id] = room
            logger.info(f"创建房间成功: {room_id}")
            return room
            
        except Exception as e:
            logger.error(f"创建房间失败: {e}")
            raise
        
    def delete_room(self, room_id: str) -> None:
        """删除房间"""
        try:
            if room_id not in self.rooms:
                raise ValueError(f"房间ID {room_id} 不存在")
                
            room = self.rooms[room_id]
            # 将所有用户从房间中移除
            for user_id in room.users.copy():
                self.leave_room(user_id, room_id)
                
            del self.rooms[room_id]
            logger.info(f"删除房间成功: {room_id}")
            
        except Exception as e:
            logger.error(f"删除房间失败: {e}")
            raise
        
    def join_room(self, user_id: str, room_id: str) -> None:
        """用户加入房间"""
        try:
            if room_id not in self.rooms:
                raise ValueError(f"房间ID {room_id} 不存在")
                
            if user_id not in self.users:
                raise ValueError(f"用户ID {user_id} 不存在")
                
            room = self.rooms[room_id]
            user = self.users[user_id]
            
            # 检查房间是否已满
            if len(room.users) >= room.max_users:
                raise ValueError(f"房间 {room_id} 已满")
                
            # 如果用户在其他房间，先离开
            if user.room_id:
                self.leave_room(user_id, user.room_id)
                
            # 加入新房间
            room.users.add(user_id)
            user.room_id = room_id
            logger.info(f"用户 {user_id} 加入房间 {room_id}")
            
        except Exception as e:
            logger.error(f"加入房间失败: {e}")
            raise
        
    def leave_room(self, user_id: str, room_id: str) -> None:
        """用户离开房间"""
        try:
            if room_id not in self.rooms:
                raise ValueError(f"房间ID {room_id} 不存在")
                
            if user_id not in self.users:
                raise ValueError(f"用户ID {user_id} 不存在")
                
            room = self.rooms[room_id]
            user = self.users[user_id]
            
            if user_id in room.users:
                room.users.remove(user_id)
                user.room_id = None
                logger.info(f"用户 {user_id} 离开房间 {room_id}")
                
            # 如果房间空了且不是永久房间，删除房间
            if not room.users:
                self.delete_room(room_id)
                
        except Exception as e:
            logger.error(f"离开房间失败: {e}")
            raise
        
    def add_user(self, user_id: str, user_name: str) -> User:
        """添加新用户"""
        try:
            if user_id in self.users:
                raise ValueError(f"用户ID {user_id} 已存在")
                
            user = User(id=user_id, name=user_name)
            self.users[user_id] = user
            logger.info(f"添加用户成功: {user_id}")
            return user
            
        except Exception as e:
            logger.error(f"添加用户失败: {e}")
            raise
        
    def remove_user(self, user_id: str) -> None:
        """移除用户"""
        try:
            if user_id not in self.users:
                raise ValueError(f"用户ID {user_id} 不存在")
                
            user = self.users[user_id]
            
            # 如果用户在房间中，先离开房间
            if user.room_id:
                self.leave_room(user_id, user.room_id)
                
            del self.users[user_id]
            logger.info(f"移除用户成功: {user_id}")
            
        except Exception as e:
            logger.error(f"移除用户失败: {e}")
            raise
        
    def get_room(self, room_id: str) -> Optional[Room]:
        """获取房间信息"""
        return self.rooms.get(room_id)
        
    def get_user(self, user_id: str) -> Optional[User]:
        """获取用户信息"""
        return self.users.get(user_id)
        
    def get_room_users(self, room_id: str) -> List[User]:
        """获取房间中的所有用户"""
        try:
            if room_id not in self.rooms:
                raise ValueError(f"房间ID {room_id} 不存在")
                
            room = self.rooms[room_id]
            return [self.users[user_id] for user_id in room.users]
            
        except Exception as e:
            logger.error(f"获取房间用户失败: {e}")
            return []
        
    def update_user_activity(self, user_id: str) -> None:
        """更新用户活动时间"""
        try:
            if user_id in self.users:
                self.users[user_id].last_active = time.time()
                
        except Exception as e:
            logger.error(f"更新用户活动时间失败: {e}")
        
    def cleanup_inactive_users(self) -> None:
        """清理不活跃用户"""
        try:
            current_time = time.time()
            
            # 检查是否需要清理
            if current_time - self.last_cleanup < self.cleanup_interval:
                return
                
            self.last_cleanup = current_time
            
            # 查找不活跃用户
            inactive_users = [
                user_id for user_id, user in self.users.items()
                if current_time - user.last_active > self.inactive_timeout
            ]
            
            # 移除不活跃用户
            for user_id in inactive_users:
                logger.info(f"清理不活跃用户: {user_id}")
                self.remove_user(user_id)
                
        except Exception as e:
            logger.error(f"清理不活跃用户失败: {e}")
        
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "rooms": {
                room_id: room.to_dict()
                for room_id, room in self.rooms.items()
            },
            "users": {
                user_id: {
                    "id": user.id,
                    "name": user.name,
                    "room_id": user.room_id,
                    "connected": user.connected,
                    "last_active": user.last_active
                }
                for user_id, user in self.users.items()
            }
        }
        
    def save_state(self, filename: str) -> None:
        """保存状态到文件"""
        try:
            state = self.to_dict()
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            logger.info(f"保存状态成功: {filename}")
            
        except Exception as e:
            logger.error(f"保存状态失败: {e}")
        
    def load_state(self, filename: str) -> None:
        """从文件加载状态"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                state = json.load(f)
                
            # 清空当前状态
            self.rooms.clear()
            self.users.clear()
            
            # 恢复用户
            for user_data in state["users"].values():
                user = User(**user_data)
                self.users[user.id] = user
                
            # 恢复房间
            for room_data in state["rooms"].values():
                room = Room(
                    id=room_data["id"],
                    name=room_data["name"],
                    owner_id=room_data["owner_id"],
                    max_users=room_data["max_users"],
                    created_at=room_data["created_at"]
                )
                room.users = set(room_data["users"])
                self.rooms[room.id] = room
                
            logger.info(f"加载状态成功: {filename}")
            
        except Exception as e:
            logger.error(f"加载状态失败: {e}")
            
    def __del__(self):
        """清理资源"""
        # 保存最终状态
        try:
            self.save_state("room_state.json")
        except:
            pass 
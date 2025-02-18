from typing import Dict, Optional
from dataclasses import dataclass
import time

@dataclass
class ParticipantStats:
    """参会者统计信息"""
    join_time: float
    last_active: float
    message_count: int
    data_sent: int
    data_received: int

class JitsiParticipant:
    """Jitsi 参会者"""
    
    def __init__(self, user_id: str, display_name: Optional[str] = None):
        self.user_id = user_id
        self.display_name = display_name or user_id
        self.roles = set()  # 角色集合
        self.permissions = set()  # 权限集合
        self._stats = ParticipantStats(
            join_time=time.time(),
            last_active=time.time(),
            message_count=0,
            data_sent=0,
            data_received=0
        )
        self._active = True

    def add_role(self, role: str):
        """添加角色"""
        self.roles.add(role)

    def remove_role(self, role: str):
        """移除角色"""
        self.roles.discard(role)

    def has_role(self, role: str) -> bool:
        """检查是否有指定角色"""
        return role in self.roles

    def grant_permission(self, permission: str):
        """授予权限"""
        self.permissions.add(permission)

    def revoke_permission(self, permission: str):
        """撤销权限"""
        self.permissions.discard(permission)

    def has_permission(self, permission: str) -> bool:
        """检查是否有指定权限"""
        return permission in self.permissions

    def update_activity(self):
        """更新活动时间"""
        self._stats.last_active = time.time()

    def record_message(self, size: int, is_sent: bool = True):
        """记录消息统计"""
        self._stats.message_count += 1
        if is_sent:
            self._stats.data_sent += size
        else:
            self._stats.data_received += size

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'user_id': self.user_id,
            'display_name': self.display_name,
            'roles': list(self.roles),
            'permissions': list(self.permissions),
            'join_time': self._stats.join_time,
            'last_active': self._stats.last_active,
            'message_count': self._stats.message_count,
            'data_sent': self._stats.data_sent,
            'data_received': self._stats.data_received,
            'active': self._active
        }

    def set_active(self, active: bool):
        """设置活动状态"""
        self._active = active

    def is_active(self) -> bool:
        """检查是否活动"""
        return self._active 
# 房间状态(RoomState)

## 功能说明
维护和管理房间的实时状态信息。

## 状态数据
```python
@dataclass
class RoomState:
    room_id: str           # 房间ID
    created_time: float    # 创建时间
    last_active: float     # 最后活跃时间
    is_locked: bool        # 是否锁定
    members: Dict[str, RoomMember]  # 成员列表
    settings: RoomSettings  # 房间设置
```

## 房间设置
```python
@dataclass
class RoomSettings:
    max_members: int = 10   # 最大成员数
    timeout: int = 30       # 超时时间(秒)
    auto_close: bool = True # 自动关闭
```

## 状态更新
```python
def update_state(self, updates: Dict[str, Any]):
    """更新房间状态
    
    Args:
        updates: 要更新的状态字段
    """
    for key, value in updates.items():
        if hasattr(self, key):
            setattr(self, key, value)
    self.last_active = time.time()
```

## 状态查询
```python
def get_room_info(self) -> Dict[str, Any]:
    """获取房间信息"""
    return {
        'id': self.room_id,
        'created': self.created_time,
        'active': self.last_active,
        'member_count': len(self.members),
        'is_locked': self.is_locked
    }
```

## 状态监控
1. 活跃度监控
   - 记录活动时间
   - 检测超时状态
   - 触发清理事件

2. 成员状态
   - 跟踪在线状态
   - 更新活跃时间
   - 维护成员列表 
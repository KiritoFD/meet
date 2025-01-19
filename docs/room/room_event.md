# 房间事件(RoomEvent)

## 功能说明
处理房间内的各类事件，包括成员状态变化、消息广播等。

## 事件类型
```python
class RoomEventType(Enum):
    # 成员事件
    MEMBER_JOIN = 'member_join'
    MEMBER_LEAVE = 'member_leave'
    MEMBER_UPDATE = 'member_update'
    
    # 房间事件
    ROOM_CREATE = 'room_create'
    ROOM_CLOSE = 'room_close'
    ROOM_UPDATE = 'room_update'
    
    # 数据事件
    POSE_UPDATE = 'pose_update'
    FRAME_UPDATE = 'frame_update'
    
    # 系统事件
    ERROR = 'error'
    WARNING = 'warning'
```

## 事件处理
```python
class RoomEventHandler:
    def handle_event(self, event_type: RoomEventType, data: Dict):
        """处理房间事件
        
        Args:
            event_type: 事件类型
            data: 事件数据
        """
        handler = self._get_handler(event_type)
        if handler:
            handler(data)
```

## 事件监听
```python
# 注册事件监听器
@room_manager.on(RoomEventType.MEMBER_JOIN)
def on_member_join(data):
    member_id = data['member_id']
    room_id = data['room_id']
    # 处理成员加入逻辑

# 注册多个事件
@room_manager.on([
    RoomEventType.POSE_UPDATE,
    RoomEventType.FRAME_UPDATE
])
def on_data_update(data):
    # 处理数据更新
```

## 事件广播
```python
# 广播事件到房间
room_manager.broadcast_event(
    RoomEventType.POSE_UPDATE,
    {
        'pose_data': pose_data,
        'timestamp': time.time()
    },
    room_id
)
```

## 错误处理
1. 事件验证
   - 检查事件类型
   - 验证数据格式
   - 权限检查

2. 错误事件
   - 发送错误通知
   - 记录错误日志
   - 错误恢复 
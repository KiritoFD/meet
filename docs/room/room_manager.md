# 房间管理器(RoomManager)

## 功能说明
负责管理视频会议房间的创建、销毁和成员管理。

## 数据结构
```python
class RoomManager:
    def __init__(self, socket: SocketManager, config: RoomConfig = None):
        self._rooms: Dict[str, Dict] = {}        # 房间字典
        self._members: Dict[str, Dict[str, RoomMember]] = {}  # 房间成员字典
        self._user_room: Dict[str, str] = {}     # 用户当前房间
```

## API说明

### create_room()
```python
def create_room(self, room_id: str) -> bool:
    """创建新房间
    
    Args:
        room_id: 房间ID
        
    Returns:
        bool: 创建是否成功
    """
```

### join_room()
```python
def join_room(self, room_id: str, user_id: str) -> bool:
    """加入房间
    
    Args:
        room_id: 房间ID
        user_id: 用户ID
        
    Returns:
        bool: 加入是否成功
    """
```

### leave_room()
```python
def leave_room(self, room_id: str, user_id: str):
    """离开房间
    
    Args:
        room_id: 房间ID
        user_id: 用户ID
    """
```

### broadcast()
```python
def broadcast(self, event: str, data: Dict[str, Any], room_id: str) -> bool:
    """广播房间消息
    
    Args:
        event: 事件名称
        data: 消息数据
        room_id: 目标房间ID
        
    Returns:
        bool: 广播是否成功
    """
```

## 房间生命周期
1. 创建阶段
   - 验证房间ID唯一性
   - 初始化房间状态
   - 创建成员列表

2. 运行阶段
   - 成员加入/离开
   - 消息广播
   - 活跃度检查

3. 清理阶段
   - 超时检测
   - 资源释放
   - 成员清理

## 错误处理
- 房间已存在
- 房间不存在
- 成员数超限
- 权限验证失败 
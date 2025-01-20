# 姿态数据发送器(PoseSender)

## 功能说明
负责将姿态数据通过Socket发送到接收端。

## 数据结构
```python
@dataclass
class PoseData:
    """姿态数据结构"""
    pose_landmarks: Optional[List[Dict[str, float]]] = None
    face_landmarks: Optional[List[Dict[str, float]]] = None
    hand_landmarks: Optional[List[Dict[str, float]]] = None
    timestamp: float = 0.0
```

## API说明

### send_pose()
```python
def send_pose(self, pose_data: PoseData, room_id: str = None) -> bool:
    """发送姿态数据
    
    Args:
        pose_data: 姿态数据
        room_id: 目标房间ID
        
    Returns:
        bool: 发送是否成功
    """
```

## 数据压缩
```python
def compress_data(self, data: PoseData) -> bytes:
    """压缩姿态数据
    
    Args:
        data: 姿态数据
        
    Returns:
        bytes: 压缩后的数据
    """
    json_str = json.dumps({
        'pose': data.pose_landmarks,
        'face': data.face_landmarks,
        'hands': data.hand_landmarks,
        'timestamp': data.timestamp
    })
    return zlib.compress(json_str.encode(), self.compression_level)
``` 
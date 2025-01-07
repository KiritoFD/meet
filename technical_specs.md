# Meeting Scene Saver - 技术规格说明

## 1. 系统架构

### 1.1 当前实现 (1对1视频)
- 基于Flask的Web服务器
- WebSocket实时通信
- MediaPipe多模态识别
  - 姿态检测 (33个关键点)
  - 手部识别 (21个关键点/手)
  - 面部网格 (468个关键点)

### 1.2 未来扩展
- 多人会议支持
- 3D模型渲染
- 场景重建
- 动作录制
- 背景替换

## 2. 数据格式

### 2.1 姿态数据
```json
{
    "status": "success",
    "pose": [[x, y, z], ...],  // 33个关键点
    "hands": [[[x, y, z], ...], [[x, y, z], ...]],  // 双手数据
    "face": [[x, y, z], ...]  // 面部网格数据
}
```

### 2.2 视频配置
```python
VIDEO_CONFIG = {
    "width": 1280,
    "height": 720,
    "fps": 30,
    "format": "MJPEG"
}
```

### 2.3 未来扩展数据格式
```python
# 房间数据
ROOM_DATA = {
    "room_id": str,
    "users": List[str],
    "status": str,
    "created_at": datetime
}

# 3D模型数据
MODEL_DATA = {
    "model_id": str,
    "vertices": List[float],
    "faces": List[int],
    "textures": List[str]
}

# 录制数据
RECORDING_DATA = {
    "recording_id": str,
    "user_id": str,
    "frames": List[dict],
    "duration": float
}
```

## 3. API接口

### 3.1 当前实现
- `/`: 主页
- `/video_feed`: 视频流
- `/pose`: 姿态数据
- `/start_capture`: 启动摄像头
- `/stop_capture`: 停止摄像头

### 3.2 未来扩展
```python
# WebSocket事件
SOCKET_EVENTS = {
    "join_room": "加入房间",
    "leave_room": "离开房间",
    "user_joined": "用户加入通知",
    "user_left": "用户离开通知",
    "pose_update": "姿态数据更新",
    "chat_message": "聊天消息"
}

# HTTP接口
FUTURE_ENDPOINTS = {
    "/api/rooms": "房间管理",
    "/api/models": "模型管理",
    "/api/recordings": "录制管理",
    "/api/users": "用户管理"
}
```

## 4. 性能指标

### 4.1 当前目标
- 视频帧率: 30fps
- 姿态检测延迟: <50ms
- 数据传输延迟: <100ms

### 4.2 未来优化
- 视频压缩优化
- 姿态数据压缩
- WebGL渲染优化
- 多线程处理

## 5. 开发路线

### 5.1 当前阶段 (1对1视频)
- [x] 基础视频流
- [x] 姿态检测
- [x] 手部识别
- [x] 面部网格
- [ ] 实时数据优化

### 5.2 下一阶段
- [ ] 多人房间
- [ ] 3D模型集成
- [ ] 动作录制
- [ ] 场景重建
- [ ] 背景处理

## 6. 依赖说明

### 6.1 核心依赖
- Python 3.8
- OpenCV 4.x
- MediaPipe 0.9.x
- Flask 2.x
- NumPy 1.x

### 6.2 未来依赖
- Three.js (3D渲染)
- Socket.IO (实时通信)
- WebRTC (点对点通信)
- TensorFlow (模型训练) 
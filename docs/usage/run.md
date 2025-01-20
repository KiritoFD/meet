# 发送端启动说明

## 功能说明
发送端程序负责采集摄像头视频、检测姿态并发送数据。

## 启动配置
```python
RUN_CONFIG = {
    'camera': {
        'device_id': 0,          # 摄像头ID
        'width': 640,            # 分辨率
        'height': 480,
        'fps': 30               # 帧率
    },
    'socket': {
        'host': 'localhost',     # 服务器地址
        'port': 5000,           # 端口
        'room_id': None         # 房间ID(可选)
    }
}
```

## 使用方法
```bash
# 基本用法
python run.py

# 指定摄像头
python run.py --camera 1

# 指定房间
python run.py --room test_room

# 指定服务器
python run.py --host example.com --port 5000
```

## 程序流程
1. 初始化组件
```python
# 初始化摄像头
camera = Camera(config.camera)

# 初始化检测器
detector = PoseDetector()

# 初始化Socket连接
socket = SocketManager(config.socket)
```

2. 主循环
```python
while True:
    # 读取视频帧
    frame = camera.read()
    
    # 检测姿态
    results = detector.detect(frame)
    
    # 发送数据
    if results:
        socket.emit('pose_update', {
            'pose_data': results,
            'timestamp': time.time()
        })
    
    # 显示预览
    if config.preview:
        drawer.draw_frame(frame, results)
        cv2.imshow('Preview', frame)
```

## 错误处理
- 摄像头异常
- 网络断开
- 检测失败 
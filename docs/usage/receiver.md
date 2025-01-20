# 接收端启动说明

## 功能说明
接收端程序负责接收姿态数据并实时显示。

## 启动配置
```python
RECEIVER_CONFIG = {
    'display': {
        'width': 640,           # 显示分辨率
        'height': 480,
        'window_name': 'Pose Receiver'
    },
    'socket': {
        'host': 'localhost',    # 服务器地址
        'port': 5000,
        'room_id': None        # 房间ID(可选)
    }
}
```

## 使用方法
```bash
# 基本用法
python receiver.py

# 指定房间
python receiver.py --room test_room

# 指定服务器
python receiver.py --host example.com --port 5000
```

## 程序流程
1. 初始化组件
```python
# 初始化显示器
display = Display(config.display)

# 初始化Socket连接
socket = SocketManager(config.socket)

# 初始化绘制器
drawer = PoseDrawer()
```

2. 事件处理
```python
# 接收姿态数据
@socket.on('pose_update')
def on_pose_update(data):
    # 解压数据
    if data.get('compressed'):
        pose_data = decompress(data['pose_data'])
    else:
        pose_data = data['pose_data']
    
    # 绘制显示
    frame = drawer.draw_frame(display.frame, pose_data)
    display.show(frame)
```

## 性能优化
1. 显示优化
   - 双缓冲显示
   - 局部更新
   - 帧率控制

2. 数据处理
   - 解压缓存
   - 异步处理
   - 帧同步 
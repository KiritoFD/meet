# 摄像头配置

## 基本配置
```python
CAMERA_CONFIG = {
    'api_preference': cv2.CAP_DSHOW,
    'device_id': 0,
    'params': {
        cv2.CAP_PROP_SETTINGS: 0,     # 禁用设置弹窗
        cv2.CAP_PROP_EXPOSURE: -3,    # 曝光值
        cv2.CAP_PROP_BRIGHTNESS: 100, # 亮度
        cv2.CAP_PROP_CONTRAST: 50,    # 对比度
        cv2.CAP_PROP_GAIN: 100,       # 增益
        cv2.CAP_PROP_AUTOFOCUS: 0,    # 禁用自动对焦
        cv2.CAP_PROP_BUFFERSIZE: 1,   # 最小缓冲
        cv2.CAP_PROP_FPS: 30,         # 帧率
        cv2.CAP_PROP_FRAME_WIDTH: 640,  # 分辨率
        cv2.CAP_PROP_FRAME_HEIGHT: 480,
        cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*'MJPG')
    }
}
```

## 重试机制
```python
'retry': {
    'max_attempts': 3,    # 最大重试次数
    'timeout': 3,         # 超时时间(秒)
    'interval': 0.1       # 重试间隔(秒)
}
```

## 性能优化
- 最小化缓冲区大小
- 禁用不必要的自动调节
- 选择合适的分辨率和帧率 
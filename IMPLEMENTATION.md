# Meeting Scene Saver - 实现参考

## 1. 已实现的核心功能

### 1.1 姿态检测 (PoseDetector)
- 使用 MediaPipe 进行姿态检测
- 实现了关键点数据的序列化和反序列化
- 支持基本的姿态验证
- 实现了简单的角度计算功能

### 1.2 音频处理 (AudioProcessor)
- 使用 PyAudio 进行音频采集
- 使用 Opus 编解码器进行音频压缩
- 支持基本的音频流处理
- 实现了音频设备管理

### 1.3 数据处理 (DataProcessor)
- 实现了姿态数据的验证和标准化
- 使用移动平均进行姿态平滑
- 支持数据包的压缩和序列化
- 实现了基本的带宽计算

### 1.4 房间管理 (RoomManager)
- 支持房间的创建和删除
- 实现了用户加入/离开房间功能
- 支持房间状态的保存和加载
- 实现了不活跃用户的自动清理

### 1.5 场景渲染 (SceneRenderer)
- 实现了基本的帧缓冲管理
- 支持简单的抗锯齿处理
- 实现了帧率计算和控制
- 支持JPEG格式的帧输出

## 2. 网络通信

### 2.1 WebSocket服务器
- 实现了基本的连接管理
- 支持房间内的消息广播
- 实现了简单的错误处理
- 支持JSON格式的消息交换

### 2.2 HTTP服务器
- 使用 Flask 框架
- 支持视频流传输
- 实现了基本的路由处理
- 支持WebSocket集成

## 3. 用户界面

### 3.1 前端实现
```html
<!-- index.html 主要结构 -->
<div class="container">
    <header>
        <h1>Meeting Scene Saver</h1>
        <div class="status-bar">
            <span id="connection-status">未连接</span>
            <span id="room-info">未加入房间</span>
        </div>
    </header>
    
    <main>
        <div class="video-container">
            <video id="local-video" autoplay muted></video>
            <div class="controls">
                <button id="start-btn">开始</button>
                <button id="stop-btn">停止</button>
                <button id="snapshot-btn">截图</button>
            </div>
        </div>
        
        <div class="info-panel">
            <div class="system-status">
                <h3>系统状态</h3>
                <div id="status-content"></div>
            </div>
            
            <div class="recording-list">
                <h3>录制列表</h3>
                <ul id="recordings"></ul>
            </div>
        </div>
    </main>
</div>
```

### 3.2 样式实现
```css
/* style.css 主要样式 */
.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

.video-container {
    position: relative;
    width: 100%;
    max-width: 800px;
    margin: 0 auto;
}

.controls {
    position: absolute;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    gap: 10px;
}

.info-panel {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-top: 20px;
}
```

### 3.3 交互实现
```javascript
// app.js 主要功能
const socket = io();
let isRecording = false;

// 连接管理
socket.on('connect', () => {
    updateStatus('已连接');
});

socket.on('disconnect', () => {
    updateStatus('未连接');
});

// 视频控制
document.getElementById('start-btn').onclick = async () => {
    try {
        const response = await fetch('/start_capture', {
            method: 'POST'
        });
        const result = await response.json();
        if (result.status === 'success') {
            startVideoStream();
        }
    } catch (error) {
        console.error('启动失败:', error);
    }
};

// 视频流处理
function startVideoStream() {
    const video = document.getElementById('local-video');
    video.src = '/video_feed';
    video.play();
}

// 状态更新
function updateStatus(status) {
    document.getElementById('connection-status').textContent = status;
}
```

## 4. 配置参考

### 4.1 音频配置
```python
@dataclass
class AudioConfig:
    channels: int = 1
    rate: int = 16000
    chunk: int = 1024
    format: int = pyaudio.paFloat32
    opus_bitrate: int = 32000
    opus_frame_size: int = 960
    noise_reduction: bool = True
    auto_gain_control: bool = True
```

### 4.2 渲染配置
```python
@dataclass
class RenderConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    background_color: Tuple[int, int, int] = (0, 0, 0)
    quality: int = 95  # JPEG压缩质量
    enable_shadows: bool = True
    enable_lighting: bool = True
    enable_antialiasing: bool = True
```

## 5. 开发注意事项

### 5.1 性能优化
- 使用缓冲区管理大量数据
- 实现数据压缩和增量更新
- 优化渲染和处理管线
- 合理使用多线程处理

### 5.2 错误处理
- 实现完整的异常捕获
- 添加详细的日志记录
- 提供用户友好的错误提示
- 实现自动恢复机制

### 5.3 资源管理
- 及时释放不需要的资源
- 实现缓存清理机制
- 控制内存和CPU使用
- 优化网络带宽使用 
# 显示页面(display.html)

## 功能说明
负责视频采集、姿态检测和数据发送。

## 页面结构
```html
<div class="container">
    <!-- 视频预览 -->
    <div class="video-container">
        <video id="videoPreview"></video>
        <canvas id="poseCanvas"></canvas>
    </div>
    
    <!-- 控制面板 -->
    <div class="control-panel">
        <button id="startBtn">Start</button>
        <button id="stopBtn">Stop</button>
        <button id="captureInitialBtn">Capture Initial</button>
    </div>
    
    <!-- 状态显示 -->
    <div class="status-panel">
        <div id="videoStreamStatus"></div>
        <div id="audioStreamStatus"></div>
    </div>
</div>
```

## JavaScript API

### 视频控制
```javascript
// 开始采集
async function startCapture() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true
    });
    videoElement.srcObject = stream;
}

// 停止采集
function stopCapture() {
    const stream = videoElement.srcObject;
    stream.getTracks().forEach(track => track.stop());
}
```

### 姿态渲染
```javascript
// 渲染姿态
function renderPose(results) {
    const canvas = document.getElementById('poseCanvas');
    const ctx = canvas.getContext('2d');
    
    // 清空画布
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // 绘制姿态
    if (results.pose) {
        drawPose(ctx, results.pose);
    }
}
```

### Socket通信
```javascript
// 发送姿态数据
socket.emit('pose_update', {
    pose_data: poseData,
    timestamp: Date.now()
});

// 接收状态更新
socket.on('status_update', (data) => {
    updateStatus(data);
});
```

## 性能优化
1. 视频处理
   - 使用requestAnimationFrame
   - 帧率控制
   - 画布缓存

2. 数据传输
   - 数据压缩
   - 批量发送
   - 防抖处理 
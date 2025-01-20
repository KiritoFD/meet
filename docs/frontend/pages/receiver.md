# 接收页面(receiver.html)

## 功能说明
负责接收姿态数据并实时渲染显示。

## 页面结构
```html
<div class="container">
    <!-- 视频显示 -->
    <div class="video-display">
        <canvas id="displayCanvas"></canvas>
    </div>
    
    <!-- 房间信息 -->
    <div class="room-info">
        <div id="roomId"></div>
        <div id="memberCount"></div>
    </div>
    
    <!-- 控制面板 -->
    <div class="control-panel">
        <button id="leaveBtn">Leave Room</button>
        <div class="volume-control">
            <input type="range" id="volumeSlider">
            <div id="volumeValue">0 dB</div>
        </div>
    </div>
</div>
```

## JavaScript API

### 数据接收
```javascript
// 接收姿态数据
socket.on('pose_update', (data) => {
    if (data.compressed) {
        data = decompress(data);
    }
    renderFrame(data);
});

// 渲染帧
function renderFrame(frameData) {
    const canvas = document.getElementById('displayCanvas');
    const ctx = canvas.getContext('2d');
    
    // 绘制背景
    ctx.drawImage(frameData.background, 0, 0);
    
    // 绘制姿态
    drawPose(ctx, frameData.pose);
}
```

### 房间管理
```javascript
// 加入房间
function joinRoom(roomId) {
    socket.emit('join_room', {
        room_id: roomId,
        user_id: userId
    });
}

// 离开房间
function leaveRoom() {
    socket.emit('leave_room', {
        room_id: currentRoom,
        user_id: userId
    });
}
```

## 性能优化
1. 渲染优化
   - 双缓冲
   - 局部更新
   - 硬件加速

2. 内存管理
   - 资源释放
   - 缓存清理
   - 内存监控 
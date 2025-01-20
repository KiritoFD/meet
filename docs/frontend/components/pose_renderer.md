# 姿态渲染组件

## 功能说明
负责将姿态数据渲染到Canvas上的可复用组件。

## 组件结构
```javascript
class PoseRenderer {
    constructor(canvas, options = {}) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.options = {
            lineWidth: 2,
            pointRadius: 3,
            colors: POSE_CONFIG.drawer.colors,
            ...options
        };
    }
    
    // 渲染姿态
    render(poseData) {
        this.clear();
        if (poseData) {
            this.drawPose(poseData);
            this.drawFace(poseData);
            this.drawHands(poseData);
        }
    }
    
    // 绘制身体姿态
    drawPose(poseData) {
        const connections = POSE_CONFIG.detector.connections;
        for (const [start, end] of connections) {
            this.drawLine(
                poseData.keypoints[start],
                poseData.keypoints[end],
                this.options.colors.body
            );
        }
    }
    
    // 绘制面部特征
    drawFace(poseData) {
        if (poseData.face_landmarks) {
            for (const connection of FACE_CONNECTIONS) {
                this.drawFaceFeature(
                    poseData.face_landmarks,
                    connection,
                    this.options.colors.face
                );
            }
        }
    }
}
```

## 渲染配置
```javascript
const rendererOptions = {
    lineWidth: 2,
    pointRadius: 3,
    colors: {
        body: '#00ff00',
        face: '#ff0000',
        hands: '#ffff00'
    },
    smoothing: true,
    antialiasing: true
};
```

## 使用示例
```javascript
// 初始化渲染器
const canvas = document.getElementById('poseCanvas');
const renderer = new PoseRenderer(canvas, rendererOptions);

// 渲染循环
function animate() {
    if (poseData) {
        renderer.render(poseData);
    }
    requestAnimationFrame(animate);
}
animate();
```

## 性能优化
1. 渲染优化
   - 使用requestAnimationFrame
   - 局部更新
   - 图层缓存

2. 绘制优化
   - 路径合并
   - 避免不必要的状态切换
   - 使用离屏Canvas 
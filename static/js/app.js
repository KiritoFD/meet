// DOM 元素
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const snapshotBtn = document.getElementById('snapshot-btn');
const statusContent = document.getElementById('status-content');
const connectionStatus = document.getElementById('connection-status');
const videoFeed = document.getElementById('video-feed');

// 状态变量
let isCapturing = false;

// 初始化
function init() {
    stopBtn.disabled = true;
    snapshotBtn.disabled = true;
    updateStatus('准备就绪');
}

// 更新状态显示
function updateStatus(message) {
    statusContent.textContent = message;
}

// 更新连接状态
function updateConnectionStatus(status) {
    connectionStatus.textContent = status;
}

// 开始捕获
async function startCapture() {
    try {
        const response = await fetch('/start_capture', {
            method: 'POST'
        });
        const data = await response.json();
        
        if (data.status === 'success') {
            isCapturing = true;
            startBtn.disabled = true;
            stopBtn.disabled = false;
            snapshotBtn.disabled = false;
            updateStatus('摄像头已启动');
            updateConnectionStatus('已连接');
            
            // 重新加载视频流
            videoFeed.src = '/video_feed?' + new Date().getTime();
        } else {
            throw new Error('启动失败');
        }
    } catch (error) {
        console.error('启动错误:', error);
        updateStatus('启动失败: ' + error.message);
    }
}

// 停止捕获
async function stopCapture() {
    try {
        const response = await fetch('/stop_capture', {
            method: 'POST'
        });
        const data = await response.json();
        
        if (data.status === 'success') {
            isCapturing = false;
            startBtn.disabled = false;
            stopBtn.disabled = true;
            snapshotBtn.disabled = true;
            updateStatus('摄像头已停止');
            updateConnectionStatus('未连接');
            
            // 清除视频流
            videoFeed.src = '';
        } else {
            throw new Error('停止失败');
        }
    } catch (error) {
        console.error('停止错误:', error);
        updateStatus('停止失败: ' + error.message);
    }
}

// 截图功能
async function takeSnapshot() {
    if (!isCapturing) return;
    
    try {
        // 创建canvas
        const canvas = document.createElement('canvas');
        canvas.width = videoFeed.videoWidth;
        canvas.height = videoFeed.videoHeight;
        
        // 绘制当前帧
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoFeed, 0, 0);
        
        // 转换为图片
        const dataUrl = canvas.toDataURL('image/jpeg');
        
        // 创建下载链接
        const link = document.createElement('a');
        link.href = dataUrl;
        link.download = `snapshot_${new Date().toISOString()}.jpg`;
        link.click();
        
        updateStatus('截图已保存');
    } catch (error) {
        console.error('截图错误:', error);
        updateStatus('截图失败: ' + error.message);
    }
}

// 获取姿态数据
async function getPoseData() {
    if (!isCapturing) return;
    
    try {
        const response = await fetch('/pose');
        const data = await response.json();
        
        if (data.status !== 'no_pose' && data.pose) {
            // TODO: 处理姿态数据
            console.log('姿态数据:', data.pose);
        }
    } catch (error) {
        console.error('获取姿态数据错误:', error);
    }
}

// 事件监听
startBtn.addEventListener('click', startCapture);
stopBtn.addEventListener('click', stopCapture);
snapshotBtn.addEventListener('click', takeSnapshot);

// 定期获取姿态数据
setInterval(getPoseData, 100);

// 初始化
init(); 
class PoseApp {
    constructor() {
        this.socket = io();
        this.videoFeed = document.getElementById('videoFeed');
        this.poseInfo = document.getElementById('poseInfo');
        this.fpsDisplay = document.getElementById('fpsDisplay');
        this.statusDisplay = document.getElementById('statusDisplay');
        
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.captureBtn = document.getElementById('captureBtn');
        
        this.bindEvents();
        this.setupSocketListeners();
    }
    
    bindEvents() {
        this.startBtn.onclick = () => this.startCapture();
        this.stopBtn.onclick = () => this.stopCapture();
        this.captureBtn.onclick = () => this.captureReference();
    }
    
    setupSocketListeners() {
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.updateStatus('已连接');
        });
        
        this.socket.on('pose_data', (data) => {
            this.updatePoseInfo(data);
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.updateStatus('已断开');
        });
    }
    
    async startCapture() {
        try {
            const response = await fetch('/start_capture', {
                method: 'POST'
            });
            const data = await response.json();
            
            if (data.success) {
                this.startBtn.disabled = true;
                this.stopBtn.disabled = false;
                this.updateStatus('运行中');
            }
        } catch (error) {
            console.error('启动失败:', error);
        }
    }
    
    async stopCapture() {
        try {
            const response = await fetch('/stop_capture', {
                method: 'POST'
            });
            const data = await response.json();
            
            if (data.success) {
                this.startBtn.disabled = false;
                this.stopBtn.disabled = true;
                this.updateStatus('已停止');
            }
        } catch (error) {
            console.error('停止失败:', error);
        }
    }
    
    async captureReference() {
        try {
            const response = await fetch('/capture_initial', {
                method: 'POST'
            });
            const data = await response.json();
            
            if (data.success) {
                console.log('参考帧已捕获');
            }
        } catch (error) {
            console.error('捕获参考帧失败:', error);
        }
    }
    
    updateStatus(status) {
        this.statusDisplay.textContent = `状态: ${status}`;
    }
    
    updatePoseInfo(data) {
        this.poseInfo.textContent = JSON.stringify(data, null, 2);
    }
    
    startStatusPolling() {
        setInterval(async () => {
            try {
                const response = await fetch('/camera_status');
                const status = await response.json();
                this.fpsDisplay.textContent = `FPS: ${status.fps.toFixed(1)}`;
            } catch (error) {
                console.error('获取状态失败:', error);
            }
        }, 1000);
    }
}

class CameraSettings {
    constructor() {
        this.brightnessSlider = document.getElementById('brightnessSlider');
        this.contrastSlider = document.getElementById('contrastSlider');
        this.exposureSlider = document.getElementById('exposureSlider');
        this.gainSlider = document.getElementById('gainSlider');
        this.resolutionSelect = document.getElementById('resolutionSelect');
        this.resetBtn = document.getElementById('resetCameraBtn');
        
        this.bindEvents();
        this.loadSettings();
    }
    
    bindEvents() {
        // 绑定滑块事件
        const sliders = [
            this.brightnessSlider,
            this.contrastSlider,
            this.exposureSlider,
            this.gainSlider
        ];
        
        sliders.forEach(slider => {
            slider.addEventListener('input', () => {
                this.updateValue(slider);
            });
            
            slider.addEventListener('change', () => {
                this.updateSetting(slider);
            });
        });
        
        // 绑定分辨率选择事件
        this.resolutionSelect.addEventListener('change', () => {
            this.updateResolution();
        });
        
        // 绑定重置按钮事件
        this.resetBtn.addEventListener('click', () => {
            this.resetSettings();
        });
    }
    
    async loadSettings() {
        try {
            const response = await fetch('/camera/settings');
            const settings = await response.json();
            
            // 更新UI
            this.brightnessSlider.value = settings.brightness;
            this.contrastSlider.value = settings.contrast;
            this.exposureSlider.value = settings.exposure;
            this.gainSlider.value = settings.gain;
            this.resolutionSelect.value = `${settings.width}x${settings.height}`;
            
            // 更新显示值
            this.updateAllValues();
            
        } catch (error) {
            console.error('加载相机设置失败:', error);
        }
    }
    
    updateValue(slider) {
        const valueSpan = document.getElementById(`${slider.id}Value`);
        if (valueSpan) {
            valueSpan.textContent = slider.value;
        }
    }
    
    async updateSetting(slider) {
        try {
            const setting = slider.id.replace('Slider', '');
            const value = parseFloat(slider.value);
            
            const response = await fetch('/camera/settings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    [setting]: value
                })
            });
            
            const result = await response.json();
            if (!result.success) {
                console.error('更新设置失败');
            }
            
        } catch (error) {
            console.error('更新设置失败:', error);
        }
    }
    
    async updateResolution() {
        try {
            const [width, height] = this.resolutionSelect.value.split('x').map(Number);
            
            const response = await fetch('/camera/settings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    width,
                    height
                })
            });
            
            const result = await response.json();
            if (!result.success) {
                console.error('更新分辨率失败');
            }
            
        } catch (error) {
            console.error('更新分辨率失败:', error);
        }
    }
    
    async resetSettings() {
        try {
            const response = await fetch('/camera/reset', {
                method: 'POST'
            });
            
            const result = await response.json();
            if (result.success) {
                await this.loadSettings();
            } else {
                console.error('重置设置失败');
            }
            
        } catch (error) {
            console.error('重置设置失败:', error);
        }
    }
    
    updateAllValues() {
        const sliders = [
            this.brightnessSlider,
            this.contrastSlider,
            this.exposureSlider,
            this.gainSlider
        ];
        
        sliders.forEach(slider => this.updateValue(slider));
    }
}

// 状态更新函数
function updateCameraStatus(isActive) {
    const cameraIndicator = document.getElementById('cameraIndicator');
    const cameraStatus = document.getElementById('cameraStatus');
    
    cameraIndicator.className = `status-indicator ${isActive ? 'active' : ''}`;
    cameraStatus.textContent = isActive ? '已启动' : '未启动';
}

function updateRoomStatus(isConnected, roomId = '') {
    const roomIndicator = document.getElementById('roomIndicator');
    const roomStatus = document.getElementById('roomStatus');
    
    roomIndicator.className = `status-indicator ${isConnected ? 'active' : ''}`;
    roomStatus.textContent = isConnected ? `已连接 (${roomId})` : '未连接';
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    const app = new PoseApp();
    const cameraSettings = new CameraSettings();
    app.startStatusPolling();
}); 
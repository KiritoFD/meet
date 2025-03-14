/**
 * 关键点流控制面板
 */
class KeypointStreamControls {
    constructor() {
        this.isRunning = false;
        this.demoMode = false;
        this.container = null;
        this.statusElem = null;
        this.startBtn = null;
        this.stopBtn = null;
        this.demoBtn = null;
        this.networkSelect = null;
        this.streamFrame = null;
        
        // 网络配置文件选项
        this.networkProfiles = [
            { id: 'high', name: '高速网络 (5Mbps)' },
            { id: 'medium', name: '中速网络 (1Mbps)' },
            { id: 'low', name: '低速网络 (300Kbps)' },
            { id: 'unstable', name: '不稳定网络' },
            { id: 'mobile', name: '移动网络 (500Kbps)' }
        ];
    }
    
    init() {
        // 创建控制面板
        this.createPanel();
        
        // 获取初始状态
        this.fetchStatus();
        
        // 每5秒更新一次状态
        setInterval(() => this.fetchStatus(), 5000);
        
        // 检查是否已有参考帧
        this.checkReferenceStatus();
    }
    
    createPanel() {
        // 创建面板容器
        this.container = document.createElement('div');
        this.container.id = 'keypoint-stream-controls';
        this.container.style.cssText = `
            position: fixed;
            bottom: 250px;
            left: 20px;
            background: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 15px;
            border-radius: 5px;
            font-family: Arial, sans-serif;
            z-index: 1000;
            width: 300px;
        `;
        
        // 创建标题
        const title = document.createElement('h3');
        title.textContent = '关键点流控制';
        title.style.cssText = 'margin: 0 0 10px 0; font-size: 16px;';
        this.container.appendChild(title);
        
        // 创建状态显示
        this.statusElem = document.createElement('div');
        this.statusElem.textContent = '状态: 加载中...';
        this.statusElem.style.cssText = 'margin-bottom: 15px; font-size: 14px;';
        this.container.appendChild(this.statusElem);
        
        // 创建流视图
        this.streamFrame = document.createElement('div');
        this.streamFrame.style.cssText = `
            width: 100%;
            height: 200px;
            background: #000;
            margin-bottom: 15px;
            border-radius: 4px;
            overflow: hidden;
            display: none;
        `;
        
        const streamImg = document.createElement('img');
        streamImg.id = 'keypoint-stream-preview';
        streamImg.style.cssText = 'width: 100%; height: 100%; object-fit: contain;';
        streamImg.alt = '关键点流预览';
        this.streamFrame.appendChild(streamImg);
        
        this.container.appendChild(this.streamFrame);
        
        // 创建网络配置选择器
        const networkLabel = document.createElement('div');
        networkLabel.textContent = '网络配置:';
        networkLabel.style.cssText = 'margin-bottom: 5px; font-size: 14px;';
        this.container.appendChild(networkLabel);
        
        this.networkSelect = document.createElement('select');
        this.networkSelect.style.cssText = `
            width: 100%;
            padding: 8px;
            margin-bottom: 15px;
            background: #333;
            color: white;
            border: 1px solid #555;
            border-radius: 4px;
        `;
        
        this.networkProfiles.forEach(profile => {
            const option = document.createElement('option');
            option.value = profile.id;
            option.textContent = profile.name;
            if (profile.id === 'medium') {
                option.selected = true;
            }
            this.networkSelect.appendChild(option);
        });
        
        this.networkSelect.addEventListener('change', () => {
            this.setNetworkProfile(this.networkSelect.value);
        });
        
        this.container.appendChild(this.networkSelect);
        
        // 创建按钮组
        const btnContainer = document.createElement('div');
        btnContainer.style.cssText = 'display: flex; gap: 10px;';
        
        // 启动按钮
        this.startBtn = document.createElement('button');
        this.startBtn.textContent = '启动流';
        this.startBtn.style.cssText = `
            padding: 8px 12px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            flex: 1;
        `;
        this.startBtn.onclick = () => this.startStream();
        btnContainer.appendChild(this.startBtn);
        
        // 停止按钮
        this.stopBtn = document.createElement('button');
        this.stopBtn.textContent = '停止流';
        this.stopBtn.style.cssText = `
            padding: 8px 12px;
            background: #F44336;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            flex: 1;
        `;
        this.stopBtn.disabled = true;
        this.stopBtn.onclick = () => this.stopStream();
        btnContainer.appendChild(this.stopBtn);
        
        this.container.appendChild(btnContainer);
        
        // 演示模式按钮
        this.demoBtn = document.createElement('button');
        this.demoBtn.textContent = '启用演示模式';
        this.demoBtn.style.cssText = `
            padding: 8px 12px;
            background: #2196F3;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            width: 100%;
            margin-top: 10px;
        `;
        this.demoBtn.onclick = () => this.toggleDemoMode();
        this.container.appendChild(this.demoBtn);
        
        // 添加到文档
        document.body.appendChild(this.container);
    }
    
    fetchStatus() {
        fetch('/keypoint_stream/status')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const status = data.status;
                    this.isRunning = status.running;
                    this.demoMode = status.demo_mode;
                    
                    // 更新状态显示
                    let statusText = `状态: ${this.isRunning ? '运行中' : '已停止'}`;
                    if (this.isRunning) {
                        statusText += `<br>演示模式: ${this.demoMode ? '已启用' : '已禁用'}`;
                        if (status.receiver && status.receiver.current_fps) {
                            statusText += `<br>FPS: ${status.receiver.current_fps.toFixed(1)}`;
                        }
                        if (status.network) {
                            statusText += `<br>带宽: ${status.network.current_usage_kbps.toFixed(1)}Kbps`;
                            statusText += `<br>丢包率: ${(status.network.packet_loss * 100).toFixed(1)}%`;
                        }
                    }
                    
                    this.statusElem.innerHTML = statusText;
                    
                    // 更新按钮状态
                    this.startBtn.disabled = this.isRunning;
                    this.stopBtn.disabled = !this.isRunning;
                    this.demoBtn.textContent = this.demoMode ? '禁用演示模式' : '启用演示模式';
                    this.demoBtn.style.background = this.demoMode ? '#FF9800' : '#2196F3';
                    
                    // 更新流预览
                    if (this.isRunning && !this.streamFrame.style.display) {
                        this.streamFrame.style.display = 'block';
                        document.getElementById('keypoint-stream-preview').src = '/keypoint_video_feed';
                    } else if (!this.isRunning && this.streamFrame.style.display !== 'none') {
                        this.streamFrame.style.display = 'none';
                        document.getElementById('keypoint-stream-preview').src = '';
                    }
                }
            })
            .catch(error => {
                console.error('获取关键点流状态失败:', error);
                this.statusElem.innerHTML = '状态: 获取失败<br>错误: ' + error.message;
            });
    }
    
    checkReferenceStatus() {
        fetch('/reference_status')
            .then(response => response.json())
            .then(data => {
                if (data.success && data.has_reference) {
                    this.startBtn.disabled = false;
                } else {
                    this.startBtn.disabled = true;
                    this.statusElem.innerHTML = '状态: 需要先捕获参考帧';
                }
            })
            .catch(error => {
                console.error('获取参考帧状态失败:', error);
            });
    }
    
    startStream() {
        this.statusElem.innerHTML = '状态: 正在启动...';
        this.startBtn.disabled = true;
        
        fetch('/keypoint_stream/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ demo_mode: this.demoMode })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                this.isRunning = true;
                this.statusElem.innerHTML = '状态: 已启动';
                this.stopBtn.disabled = false;
                
                // 显示流预览
                this.streamFrame.style.display = 'block';
                document.getElementById('keypoint-stream-preview').src = '/keypoint_video_feed';
                
                // 更新状态
                this.fetchStatus();
            } else {
                this.statusElem.innerHTML = `状态: 启动失败<br>错误: ${data.error || '未知错误'}`;
                this.startBtn.disabled = false;
            }
        })
        .catch(error => {
            console.error('启动关键点流失败:', error);
            this.statusElem.innerHTML = '状态: 启动失败<br>错误: ' + error.message;
            this.startBtn.disabled = false;
        });
    }
    
    stopStream() {
        this.statusElem.innerHTML = '状态: 正在停止...';
        this.stopBtn.disabled = true;
        
        fetch('/keypoint_stream/stop', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                this.isRunning = false;
                this.statusElem.innerHTML = '状态: 已停止';
                this.startBtn.disabled = false;
                
                // 隐藏流预览
                this.streamFrame.style.display = 'none';
                document.getElementById('keypoint-stream-preview').src = '';
            } else {
                this.statusElem.innerHTML = `状态: 停止失败<br>错误: ${data.error || '未知错误'}`;
                this.stopBtn.disabled = false;
            }
        })
        .catch(error => {
            console.error('停止关键点流失败:', error);
            this.statusElem.innerHTML = '状态: 停止失败<br>错误: ' + error.message;
            this.stopBtn.disabled = false;
        });
    }
    
    toggleDemoMode() {
        const enable = !this.demoMode;
        
        fetch('/keypoint_stream/demo', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ enable })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                this.demoMode = data.demo_mode;
                this.demoBtn.textContent = this.demoMode ? '禁用演示模式' : '启用演示模式';
                this.demoBtn.style.background = this.demoMode ? '#FF9800' : '#2196F3';
                
                // 如果流已经在运行，需要重启流应用新设置
                if (this.isRunning) {
                    this.statusElem.innerHTML = '状态: 正在重启流...';
                    this.stopStream();
                    setTimeout(() => this.startStream(), 1000);
                }
            }
        })
        .catch(error => {
            console.error('切换演示模式失败:', error);
        });
    }
    
    setNetworkProfile(profile) {
        fetch('/network/set_profile', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ profile })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                this.statusElem.innerHTML = `状态: 已设置网络为 ${profile}`;
                
                // 更新状态
                setTimeout(() => this.fetchStatus(), 1000);
            } else {
                this.statusElem.innerHTML = `状态: 设置网络失败<br>错误: ${data.error || '未知错误'}`;
            }
        })
        .catch(error => {
            console.error('设置网络配置失败:', error);
            this.statusElem.innerHTML = '状态: 设置网络失败<br>错误: ' + error.message;
        });
    }
}

// 页面加载完成后初始化控制面板
document.addEventListener('DOMContentLoaded', function() {
    const controls = new KeypointStreamControls();
    controls.init();
});

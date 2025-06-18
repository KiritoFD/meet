/**
 * NVIDIA模型控制面板
 */
class NVIDIAControls {
    constructor() {
        this.isInitialized = false;
        this.isEnabled = false;
        this.container = null;
        this.statusElem = null;
        this.toggleBtn = null;
        this.initBtn = null;
    }
    
    init() {
        // 创建控制面板
        this.createPanel();
        
        // 获取初始状态
        this.fetchStatus();
        
        // 每5秒更新一次状态
        setInterval(() => this.fetchStatus(), 5000);
    }
    
    createPanel() {
        // 创建面板容器
        this.container = document.createElement('div');
        this.container.id = 'nvidia-controls';
        this.container.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 15px;
            border-radius: 5px;
            font-family: Arial, sans-serif;
            z-index: 1000;
            width: 250px;
        `;
        
        // 创建标题
        const title = document.createElement('h3');
        title.textContent = 'NVIDIA 动画控制';
        title.style.cssText = 'margin: 0 0 10px 0; font-size: 16px;';
        this.container.appendChild(title);
        
        // 创建状态显示
        this.statusElem = document.createElement('div');
        this.statusElem.textContent = '状态: 加载中...';
        this.statusElem.style.cssText = 'margin-bottom: 15px; font-size: 14px;';
        this.container.appendChild(this.statusElem);
        
        // 创建按钮容器
        const btnContainer = document.createElement('div');
        btnContainer.style.cssText = 'display: flex; gap: 10px;';
        
        // 初始化按钮
        this.initBtn = document.createElement('button');
        this.initBtn.textContent = '初始化模型';
        this.initBtn.style.cssText = `
            padding: 8px 12px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            flex: 1;
        `;
        this.initBtn.onclick = () => this.initializeModel();
        btnContainer.appendChild(this.initBtn);
        
        // 切换按钮
        this.toggleBtn = document.createElement('button');
        this.toggleBtn.textContent = '启用模型';
        this.toggleBtn.style.cssText = `
            padding: 8px 12px;
            background: #2196F3;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            flex: 1;
        `;
        this.toggleBtn.disabled = true;
        this.toggleBtn.onclick = () => this.toggleModel();
        btnContainer.appendChild(this.toggleBtn);
        
        this.container.appendChild(btnContainer);
        
        // 添加到文档
        document.body.appendChild(this.container);
    }
    
    fetchStatus() {
        fetch('/nvidia/status')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const status = data.status;
                    this.isInitialized = status.initialized;
                    
                    // 更新状态显示
                    let statusText = `状态: ${this.isInitialized ? '已初始化' : '未初始化'}`;
                    if (this.isInitialized) {
                        statusText += `<br>设备: ${status.device}`;
                    }
                    if (status.has_source_keypoints) {
                        statusText += '<br>已加载参考帧';
                    }
                    
                    this.statusElem.innerHTML = statusText;
                    
                    // 更新按钮状态
                    this.initBtn.disabled = this.isInitialized;
                    this.toggleBtn.disabled = !this.isInitialized;
                    
                    // 获取启用状态
                    this.fetchEnabledStatus();
                }
            })
            .catch(error => {
                console.error('获取NVIDIA状态失败:', error);
                this.statusElem.textContent = '状态: 获取失败';
            });
    }
    
    fetchEnabledStatus() {
        fetch('/check_stream_status')
            .then(response => response.json())
            .then(data => {
                if (data.nvidia_model) {
                    this.isEnabled = data.nvidia_model.enabled;
                    this.toggleBtn.textContent = this.isEnabled ? '禁用模型' : '启用模型';
                    this.toggleBtn.style.background = this.isEnabled ? '#F44336' : '#2196F3';
                }
            })
            .catch(error => {
                console.error('获取流状态失败:', error);
            });
    }
    
    initializeModel() {
        this.statusElem.textContent = '状态: 正在初始化...';
        this.initBtn.disabled = true;
        
        fetch('/nvidia/initialize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                this.isInitialized = true;
                this.statusElem.textContent = '状态: 已初始化';
                this.toggleBtn.disabled = false;
                
                // 自动启用模型
                this.toggleModel(true);
            } else {
                this.statusElem.textContent = `状态: 初始化失败 - ${data.error || '未知错误'}`;
                this.initBtn.disabled = false;
            }
        })
        .catch(error => {
            console.error('初始化NVIDIA模型失败:', error);
            this.statusElem.textContent = '状态: 初始化失败';
            this.initBtn.disabled = false;
        });
    }
    
    toggleModel(forceEnable = null) {
        const enable = forceEnable !== null ? forceEnable : !this.isEnabled;
        this.toggleBtn.disabled = true;
        
        fetch('/nvidia/toggle', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ enable })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                this.isEnabled = data.enabled;
                this.toggleBtn.textContent = this.isEnabled ? '禁用模型' : '启用模型';
                this.toggleBtn.style.background = this.isEnabled ? '#F44336' : '#2196F3';
            }
            this.toggleBtn.disabled = false;
        })
        .catch(error => {
            console.error('切换NVIDIA模型失败:', error);
            this.toggleBtn.disabled = false;
        });
    }
}

// 页面加载完成后初始化控制面板
document.addEventListener('DOMContentLoaded', function() {
    const controls = new NVIDIAControls();
    controls.init();
});

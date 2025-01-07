class PoseRecorder {
    constructor() {
        this.isCalibrating = false;
        this.isRecording = false;
        this.currentPoseIndex = 0;
        this.poses = ['T', 'A', 'N'];
        this.poseNames = {
            'T': 'T-pose (双手平举)',
            'A': 'A-pose (双手45度)',
            'N': '自然站姿'
        };
        
        // DOM元素
        this.calibrateButton = document.getElementById('calibrateButton');
        this.recordButton = document.getElementById('recordButton');
        this.calibrationGuide = document.getElementById('calibrationGuide');
        this.poseGuide = document.getElementById('poseGuide');
        this.calibrationProgress = document.getElementById('calibrationProgress');
        
        // 绑定事件处理器
        this.calibrateButton.onclick = () => this.startCalibration();
        this.recordButton.onclick = () => this.startRecording();
        
        // 初始化
        this.recordButton.disabled = true;
    }
    
    async startCalibration() {
        try {
            this.isCalibrating = true;
            this.currentPoseIndex = 0;
            
            // 显示校准指引
            this.calibrationGuide.style.display = 'block';
            this.updatePoseGuide();
            
            // 禁用按钮
            this.recordButton.disabled = true;
            this.calibrateButton.disabled = true;
            
            // 开始校准流程
            await this.nextCalibrationPose();
            
        } catch (error) {
            console.error('开始校准失败:', error);
            this.showMessage('校准失败', 'error');
        }
    }
    
    async nextCalibrationPose() {
        if (this.currentPoseIndex >= this.poses.length) {
            // 校准完成
            await this.finishCalibration();
            return;
        }
        
        const pose = this.poses[this.currentPoseIndex];
        this.updatePoseGuide();
        
        // 等待3秒让用户做好准备
        await this.countdown(3);
        
        // 捕获姿势数据
        try {
            const response = await fetch('/api/calibration', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    pose_type: pose
                })
            });
            
            const data = await response.json();
            if (data.status === 'success') {
                this.currentPoseIndex++;
                this.updateCalibrationProgress();
                
                if (this.currentPoseIndex < this.poses.length) {
                    // 继续下一个姿势
                    setTimeout(() => this.nextCalibrationPose(), 1000);
                } else {
                    // 完成校准
                    await this.finishCalibration();
                }
            } else {
                throw new Error(data.message);
            }
            
        } catch (error) {
            console.error('校准姿势失败:', error);
            this.showMessage('校准姿势失败', 'error');
        }
    }
    
    updatePoseGuide() {
        this.poseGuide.innerHTML = this.poses.map((pose, index) => `
            <li class="${index === this.currentPoseIndex ? 'active' : index < this.currentPoseIndex ? 'done' : ''}">
                ${this.poseNames[pose]}
            </li>
        `).join('');
    }
    
    updateCalibrationProgress() {
        const progress = (this.currentPoseIndex / this.poses.length) * 100;
        this.calibrationProgress.style.width = `${progress}%`;
    }
    
    async finishCalibration() {
        this.isCalibrating = false;
        this.calibrationGuide.style.display = 'none';
        this.recordButton.disabled = false;
        this.calibrateButton.disabled = false;
        this.showMessage('校准完成');
    }
    
    async startRecording() {
        try {
            const response = await fetch('/api/recording/start', {
                method: 'POST'
            });
            
            const data = await response.json();
            if (data.status === 'success') {
                this.isRecording = true;
                this.recordButton.textContent = '停止录制';
                this.recordButton.onclick = () => this.stopRecording();
                this.showMessage('开始录制');
            } else {
                throw new Error(data.message);
            }
            
        } catch (error) {
            console.error('开始录制失败:', error);
            this.showMessage('开始录制失败', 'error');
        }
    }
    
    async stopRecording() {
        try {
            const response = await fetch('/api/recording/stop', {
                method: 'POST'
            });
            
            const data = await response.json();
            if (data.status === 'success') {
                this.isRecording = false;
                this.recordButton.textContent = '录制模型';
                this.recordButton.onclick = () => this.startRecording();
                this.showMessage('录制完成');
                
                // 触发录制完成事件
                this.dispatchEvent('recordingComplete', data.data);
            } else {
                throw new Error(data.message);
            }
            
        } catch (error) {
            console.error('停止录制失败:', error);
            this.showMessage('停止录制失败', 'error');
        }
    }
    
    // 辅助方法
    async countdown(seconds) {
        return new Promise(resolve => {
            let remaining = seconds;
            const timer = setInterval(() => {
                if (remaining <= 0) {
                    clearInterval(timer);
                    resolve();
                } else {
                    this.showMessage(`准备开始: ${remaining}秒`);
                    remaining--;
                }
            }, 1000);
        });
    }
    
    showMessage(message, type = 'info') {
        // 触发消息事件
        this.dispatchEvent('message', { message, type });
    }
    
    // 事件处理
    dispatchEvent(name, detail) {
        const event = new CustomEvent(name, { detail });
        window.dispatchEvent(event);
    }
}

export default PoseRecorder; 
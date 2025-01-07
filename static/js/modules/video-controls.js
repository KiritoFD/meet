class VideoControls {
    constructor(video, renderManager) {
        this.video = video;
        this.renderManager = renderManager;
        
        // 控制状态
        this.state = {
            isPlaying: false,
            volume: 1.0,
            opacity: 1.0
        };
        
        // 绑定控制事件
        this.bindControls();
    }
    
    bindControls() {
        // 播放/暂停
        document.getElementById('playPauseBtn').onclick = () => {
            this.togglePlay();
        };
        
        // 透明度控制
        document.getElementById('opacitySlider').oninput = (e) => {
            this.setOpacity(e.target.value);
        };
        
        // 音量控制
        document.getElementById('volumeSlider').oninput = (e) => {
            this.setVolume(e.target.value);
        };
    }
    
    togglePlay() {
        if (this.video.paused) {
            this.video.play();
            this.state.isPlaying = true;
        } else {
            this.video.pause();
            this.state.isPlaying = false;
        }
    }
    
    setOpacity(value) {
        this.state.opacity = value;
        this.renderManager.setVideoOpacity(value);
    }
    
    setVolume(value) {
        this.state.volume = value;
        this.video.volume = value;
    }
}

export default VideoControls; 
import ModelManager from './modules/model-manager.js';
import RenderManager from './modules/render-manager.js';
import PoseRecorder from './modules/pose-recorder.js';

// 初始化管理器
let modelManager;
let renderManager;
let poseRecorder;

async function init() {
    try {
        // 初始化渲染管理器
        const canvas = document.getElementById('renderCanvas');
        renderManager = new RenderManager(canvas);
        
        // 初始化视频
        const video = document.getElementById('videoPreview');
        await initVideo(video);
        
        // 绑定事件
        bindEvents();
        
        // 开始渲染循环
        animate();
        
    } catch (error) {
        console.error('初始化失败:', error);
    }
}

async function initVideo(video) {
    try {
        // 等待视频元数据加载
        await new Promise((resolve) => {
            video.addEventListener('loadedmetadata', resolve, { once: true });
        });
        
        // 初始化视频渲染
        renderManager.initVideo(video);
        
    } catch (error) {
        console.error('视频初始化失败:', error);
    }
}

function bindEvents() {
    // 监听姿态更新
    window.addEventListener('poseUpdate', (e) => {
        renderManager.updatePose(e.detail);
    });
    
    // 监听模型加载
    window.addEventListener('modelLoaded', (e) => {
        renderManager.setModel(e.detail);
    });
    
    // 监听渲染设置更新
    document.getElementById('renderQuality').addEventListener('change', (e) => {
        renderManager.setQuality(e.target.value);
    });
    
    // 1. 视频流事件
    const video = document.getElementById('videoPreview');
    video.addEventListener('play', () => {
        renderManager.initVideoTexture();
        renderManager.createVideoPlane();
    });
    
    // 2. 渲染循环中的画面更新
    function animate() {
        renderManager.updateVideoFrame();
        renderManager.render();
        requestAnimationFrame(animate);
    }
}

function animate() {
    renderManager.render();
    requestAnimationFrame(animate);
}

// 添加窗口大小调整处理
window.addEventListener('resize', () => {
    const canvas = document.getElementById('renderCanvas');
    renderManager.resize(
        canvas.clientWidth,
        canvas.clientHeight
    );
});

// 启动应用
init(); 
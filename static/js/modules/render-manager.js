import VideoTextureManager from './video-texture-manager.js';

class RenderManager {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.renderer = new THREE.WebGLRenderer({ 
            canvas,
            antialias: true,
            alpha: true
        });
        
        // 渲染配置
        this.config = {
            width: canvas.width,
            height: canvas.height,
            fps: 30,
            quality: 'high',
            enableShadows: true,
            enableAntialiasing: true
        };
        
        // 初始化场景
        this.initScene();
        
        // 绑定UI控制
        this.bindControls();
        
        // 添加视频管理器
        this.videoManager = null;
        
        // 添加渲染状态
        this.renderState = {
            isVideoReady: false,
            isModelReady: false,
            isShadowsEnabled: true
        };
    }
    
    initScene() {
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(
            75,
            this.canvas.width / this.canvas.height,
            0.1,
            1000
        );
        
        // 设置光照
        this.setupLights();
        
        // 设置后处理
        this.setupPostProcessing();
    }
    
    setupLights() {
        // 环境光
        const ambientLight = new THREE.AmbientLight(0x404040);
        this.scene.add(ambientLight);
        
        // 主光源
        const mainLight = new THREE.DirectionalLight(0xffffff, 1);
        mainLight.position.set(5, 5, 5);
        mainLight.castShadow = true;
        this.scene.add(mainLight);
    }
    
    setupPostProcessing() {
        this.composer = new THREE.EffectComposer(this.renderer);
        
        // 添加基础渲染通道
        const renderPass = new THREE.RenderPass(this.scene, this.camera);
        this.composer.addPass(renderPass);
        
        // 添加后处理效果
        if (this.config.enableAntialiasing) {
            const fxaaPass = new THREE.ShaderPass(THREE.FXAAShader);
            this.composer.addPass(fxaaPass);
        }
    }
    
    updatePose(poseData) {
        // 更新模型姿态
        if (this.currentModel && this.currentModel.skeleton) {
            this.updateSkeleton(this.currentModel.skeleton, poseData);
        }
    }
    
    updateSkeleton(skeleton, poseData) {
        // 映射姿态数据到骨骼
        const boneMap = this.getBoneMapping();
        for (const [poseLandmark, boneName] of Object.entries(boneMap)) {
            const bone = skeleton.getBoneByName(boneName);
            if (bone && poseData[poseLandmark]) {
                this.updateBonePose(bone, poseData[poseLandmark]);
            }
        }
        skeleton.update();
    }
    
    initVideo(video) {
        this.videoManager = new VideoTextureManager(video, this.renderer);
        this.videoManager.init();
        this.scene.add(this.videoManager.plane);
        
        // 设置相机位置
        this.camera.position.z = 10;
        
        this.renderState.isVideoReady = true;
    }
    
    updateVideo() {
        if (this.videoManager) {
            this.videoManager.update();
        }
    }
    
    setVideoOpacity(value) {
        if (this.videoManager) {
            this.videoManager.setOpacity(value);
        }
    }
    
    resize(width, height) {
        // 更新渲染器尺寸
        this.renderer.setSize(width, height);
        
        // 更新相机宽高比
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        
        // 更新视频平面尺寸
        if (this.videoManager) {
            this.videoManager.resize(width, height);
        }
        
        // 更新后处理效果尺寸
        if (this.composer) {
            this.composer.setSize(width, height);
        }
    }
    
    render() {
        // 更新视频纹理
        this.updateVideo();
        
        // 渲染场景
        if (this.config.quality === 'high') {
            this.composer.render();
        } else {
            this.renderer.render(this.scene, this.camera);
        }
    }
}

export default RenderManager; 
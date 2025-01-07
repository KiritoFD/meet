class VideoTextureManager {
    constructor(video, renderer) {
        this.video = video;
        this.renderer = renderer;
        this.texture = null;
        this.plane = null;
        this.material = null;
        
        // 视频配置
        this.config = {
            width: 1280,
            height: 720,
            aspectRatio: 16/9
        };
    }
    
    init() {
        // 创建视频纹理
        this.texture = new THREE.VideoTexture(this.video);
        this.texture.minFilter = THREE.LinearFilter;
        this.texture.magFilter = THREE.LinearFilter;
        
        // 创建材质
        this.material = new THREE.MeshBasicMaterial({
            map: this.texture,
            side: THREE.DoubleSide,
            transparent: true
        });
        
        // 创建平面
        const geometry = new THREE.PlaneGeometry(
            this.config.width / 100,
            this.config.height / 100
        );
        this.plane = new THREE.Mesh(geometry, this.material);
        
        // 设置初始位置
        this.plane.position.z = -5;
    }
    
    update() {
        if (this.texture && this.video.readyState >= this.video.HAVE_CURRENT_DATA) {
            this.texture.needsUpdate = true;
        }
    }
    
    setOpacity(value) {
        if (this.material) {
            this.material.opacity = value;
        }
    }
    
    resize(width, height) {
        if (this.plane) {
            this.plane.scale.set(
                width / 100,
                height / 100,
                1
            );
        }
    }
}

export default VideoTextureManager; 
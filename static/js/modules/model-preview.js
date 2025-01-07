class ModelPreview {
    constructor(container) {
        this.container = container;
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        
        this.init();
    }
    
    init() {
        // 初始化场景
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.container.appendChild(this.renderer.domElement);
        
        // 添加灯光
        const light = new THREE.DirectionalLight(0xffffff, 1);
        light.position.set(0, 1, 1);
        this.scene.add(light);
        
        // 设置相机位置
        this.camera.position.z = 5;
        
        // 开始渲染循环
        this.animate();
    }
    
    loadModel(modelData) {
        // 清除现有模型
        this.clearModel();
        
        // 创建几何体
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(modelData.vertices, 3));
        geometry.setIndex(modelData.faces);
        
        // 创建材质
        const material = new THREE.MeshPhongMaterial({
            color: 0x808080,
            wireframe: false
        });
        
        // 创建模型
        this.model = new THREE.Mesh(geometry, material);
        this.scene.add(this.model);
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        if (this.model) {
            this.model.rotation.y += 0.01;
        }
        
        this.renderer.render(this.scene, this.camera);
    }
} 
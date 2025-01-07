class ModelManager {
    constructor() {
        this.models = new Map(); // 存储已加载的模型
        this.currentModel = null;
        
        // 初始化UI元素
        this.modelList = document.getElementById('modelList');
        this.modelUploadBtn = document.getElementById('modelUploadBtn');
        
        // 绑定事件
        this.modelUploadBtn.addEventListener('click', () => this.handleModelUpload());
        
        // 初始化文件上传
        this.initFileUpload();
    }
    
    initFileUpload() {
        this.fileInput = document.createElement('input');
        this.fileInput.type = 'file';
        this.fileInput.accept = '.glb,.gltf,.fbx';
        this.fileInput.style.display = 'none';
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        document.body.appendChild(this.fileInput);
    }
    
    async handleModelUpload() {
        this.fileInput.click();
    }
    
    async handleFileSelect(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        try {
            const modelData = await this.loadModelFile(file);
            await this.addModel(modelData);
            this.updateModelList();
        } catch (error) {
            console.error('模型加载失败:', error);
        }
    }
    
    async loadModelFile(file) {
        // 根据文件类型选择加载器
        const loader = this.getModelLoader(file.name);
        return new Promise((resolve, reject) => {
            loader.load(
                URL.createObjectURL(file),
                (model) => resolve(this.processModel(model)),
                null,
                reject
            );
        });
    }
    
    getModelLoader(filename) {
        const ext = filename.split('.').pop().toLowerCase();
        switch (ext) {
            case 'glb':
            case 'gltf':
                return new THREE.GLTFLoader();
            case 'fbx':
                return new THREE.FBXLoader();
            default:
                throw new Error('不支持的文件格式');
        }
    }
    
    processModel(model) {
        // 处理模型数据
        return {
            id: Date.now().toString(),
            name: model.name || '未命名模型',
            object: model,
            skeleton: this.extractSkeleton(model),
            animations: model.animations || []
        };
    }
    
    extractSkeleton(model) {
        // 提取骨骼数据
        let skeleton = null;
        model.traverse((node) => {
            if (node.isSkeleton) {
                skeleton = node;
            }
        });
        return skeleton;
    }
}

export default ModelManager; 
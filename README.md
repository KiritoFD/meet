# Meeting Scene Saver
基于姿态捕获的超低带宽视频会议系统。通过实时捕获人体姿态并在远端重建,实现卫星级别带宽(约10kb/s)下的流畅视频会议。

现有的视频会议都是采用视频传输（连虚拟背景也是），我们的项目实现了原理的革新，预先传输模型，在视频通话中只传输捕捉到的人体姿态数据，大大降低了带宽需求。

在2D环境下，带宽降低或许并不必要（当然，我们降低到了卫星通讯水平，可能在极端环境下有用），但如果是未来的VR,AR通话，必然不可能实时传输整个3D环境。
我们的预传模型+实时姿态再渲染的技术在平面视频会议时代或许并无太大优势，但却是未来必然的选择和发展方向

## 核心特性
- 超低带宽传输: 仅传输姿态数据(~10kb/s)
- 高质量重建: 远端基于姿态数据重建人物视频
- 实时性能: 端到端延迟<200ms
- 多模态融合:
  - 人体姿态(33点)
  - 手部动作(21点/手)
  - 面部表情(468点)
- 渲染增强:
  - 光照重建
  - 材质优化
  - 后期特效

## 项目结构

```
project/
├── run.py                # 主程序入口
├── requirements.txt      # Python依赖
├── package.json         # Node.js依赖
│
├── server/              # 后端代码
│   ├── __init__.py
│   ├── model_manager.py  # 模型管理
│   ├── video_renderer.py # 视频渲染
│   ├── pose_detector.py  # 姿态检测
│   ├── scene_renderer.py # 场景渲染
│   └── data_processor.py # 数据处理
│
├── static/              # 前端资源
│   ├── js/
│   │   ├── app.js      # 应用入口
│   │   └── modules/    # 功能模块
│   │       ├── model-manager.js     # 模型管理
│   │       ├── render-manager.js    # 渲染管理
│   │       ├── pose-recorder.js     # 姿态录制
│   │       ├── video-texture-manager.js # 视频纹理
│   │       ├── video-controls.js    # 视频控制
│   │       └── scene-composer.js    # 场景合成
│   │
│   ├── css/            # 样式文件
│   │   └── style.css
│   │
│   ├── models/         # 3D模型文件
│   │   ├── default/    # 默认模型
│   │   └── uploads/    # 上传模型
│   │
│   └── shaders/        # 着色器文件
│       ├── vertex/     # 顶点着色器
│       └── fragment/   # 片段着色器
│
├── templates/           # 模板文件
│   └── index.html      # 主页面
│
└── uploads/            # 上传文件存储
    ├── models/         # 模型文件
    └── videos/         # 视频文件
```

## 环境配置

### 1. 系统要求
- CUDA 11.0+ (推荐)
- OpenGL 4.3+
- WebGL 2.0 支持的浏览器
- 8GB+ 内存

### 2. Conda环境配置
```bash
# 创建conda环境
conda create -n meeting python=3.8

# 激活环境
conda activate meeting

# 安装基础依赖
conda install -c conda-forge \
    numpy=1.23.5 \
    opencv=4.7.0 \
    flask=2.0.1 \
    pytorch=1.9.0 \
    cudatoolkit=11.0

# 安装其他依赖
pip install -r requirements.txt
```

### 3. Node.js环境配置
```bash
# 安装Node.js (推荐使用nvm)
nvm install 14
nvm use 14

# 安装依赖
npm install

# 开发模式
npm run dev

# 构建
npm run build
```

### 4. 开发服务器启动
```bash
# 激活conda环境
conda activate meeting

# 启动Flask服务器
python run.py
```

## 开发任务 (TODO)

### 1. 基础架构
- [ ] 项目结构优化
  - [ ] 模块化重构
  - [ ] 依赖管理优化
  - [ ] 构建流程优化
- [ ] 开发环境配置
  - [ ] 热重载支持
  - [ ] 调试工具集成
  - [ ] 测试框架搭建

### 2. 前端功能
- [ ] 视频渲染系统
  - [ ] WebGL渲染管线
  - [ ] 视频纹理处理
  - [ ] 后处理效果
- [ ] 模型系统
  - [ ] 模型加载器
  - [ ] 骨骼动画
  - [ ] 材质系统
- [ ] UI系统
  - [ ] 控制面板
  - [ ] 预览窗口
  - [ ] 参数调节

### 3. 后端功能
- [ ] 视频处理
  - [ ] 实时编码
  - [ ] 格式转换
  - [ ] 质量控制
- [ ] 姿态检测
  - [ ] 实时识别
  - [ ] 数据平滑
  - [ ] 动作分析
- [ ] 数据管理
  - [ ] 模型存储
  - [ ] 缓存系统
  - [ ] 会话管理

### 4. 性能优化
- [ ] 渲染优化
  - [ ] GPU加速
  - [ ] 内存管理
  - [ ] 帧率控制
- [ ] 数据优化
  - [ ] 压缩算法
  - [ ] 缓存策略
  - [ ] 异步加载

### 5. 测试计划
- [ ] 单元测试
  - [ ] 前端模块测试
  - [ ] 后端接口测试
  - [ ] 渲染测试
- [ ] 性能测试
  - [ ] 压力测试
  - [ ] 内存泄漏检测
  - [ ] 渲染性能分析

## 依赖说明

### Python依赖 (requirements.txt)
```
# Web框架
flask==2.0.1
flask-socketio==5.1.1
werkzeug==2.0.1
gunicorn==20.1.0

# 视频处理
opencv-python==4.7.0
mediapipe==0.9.0
ffmpeg-python==0.2.0
av==9.3.0

# 机器学习
torch==1.9.0
torchvision==0.10.0
numpy==1.23.5
scipy==1.7.1
scikit-learn==0.24.2

# 数据处理
pandas==1.3.3
pillow==8.3.2
h5py==3.4.0

# 实时通信
python-socketio==5.4.0
eventlet==0.33.0
websockets==10.0

# 开发工具
pytest==6.2.5
black==21.9b0
flake8==3.9.2
mypy==0.910
```

### JavaScript依赖 (package.json)
```json
{
  "dependencies": {
    "three": "^0.137.0",
    "socket.io-client": "^4.0.1",
    "@tensorflow/tfjs": "^3.9.0",
    "gl-matrix": "^3.4.3",
    "stats.js": "^0.17.0"
  },
  "devDependencies": {
    "vite": "^2.7.2",
    "eslint": "^8.6.0",
    "prettier": "^2.5.1",
    "typescript": "^4.5.4",
    "@types/three": "^0.137.0",
    "jest": "^27.4.7"
  }
}
```

## 开发规范

### 1. 代码风格
- Python: PEP 8
- JavaScript: ESLint (Airbnb)
- CSS: BEM命名

### 2. 提交规范
- feat: 新功能
- fix: 修复
- docs: 文档
- style: 格式
- refactor: 重构
- test: 测试
- chore: 构建

### 3. 分支管理
- main: 主分支
- develop: 开发分支
- feature/*: 功能分支
- bugfix/*: 修复分支

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

## 快速开始

### 环境配置
### 环境配置

1. 解压环境：
先安装anaconda https://www.anaconda.com/download/success 下载

win+r 打开cmd ,输入自己的anaconda安装路径

    cd anaconda3\envs

      (若是windows环境)：mkdir meet6

      (若linux或wsl) mkdir -p meet6

    cd meet6
复制当前路径

用anaconda prompt 进入项目根目录，输入以下命令：
(本压缩包目前在dev分支，如果发现找不到压缩包的切换到dev分支下载一下)

    tar -xvf meet6.tar.gz -C 
在这一行后粘贴刚才复制的路径，enter

    conda activate meet6
运行代码时解释器选择meet6环境

    cd meet

运行run.py
程序运行后，打开浏览器访问 `http://127.0.0.1:5000/`。

## 项目结构

```
project/
├── run.py                # 主程序入口
├── requirements.txt      # Python依赖
│
├── config/              # 配置文件
│   ├── __init__.py
│   ├── settings.py      # 全局配置
│   └── config.yaml      # YAML配置文件
│
├── camera/             # 摄像头模块
│   ├── __init__.py
│   └── manager.py      # 摄像头管理
│
├── pose/               # 姿态检测模块
│   ├── __init__.py
│   ├── detector.py     # 姿态检测器
│   ├── drawer.py       # 姿态绘制
│   ├── smoother.py     # 平滑处理
│   ├── binding.py      # 姿态绑定
│   ├── deformer.py     # 图像变形处理
│   ├── types.py        # 类型定义
│   └── pose_binding.py # 姿态绑定实现
│
├── connect/            # 连接模块
│   ├── __init__.py
│   ├── socket_manager.py  # Socket管理
│   ├── pose_sender.py    # 姿态数据发送
│   ├── pose_protocol.py  # 数据协议
│   └── room_manager.py   # 房间管理
│
├── receive/            # 接收端模块
│   ├── __init__.py
│   ├── app.py          # 接收端应用
│   ├── manager.py      # 接收端管理
│   └── static.py       # 静态资源处理
│
├── audio/              # 音频模块
│   ├── __init__.py
│   └── processor.py    # 音频处理
│
├── tools/              # 工具脚本
│   └── demo.py         # 演示程序
│
├── docs/               # 文档
│   ├── README.md       # 文档索引
│   ├── config/         # 配置文档
│   ├── testing/        # 测试文档
│   ├── usage/          # 使用说明
│   ├── pose/           # 姿态模块文档
│   ├── room/           # 房间管理文档
│   ├── receive/        # 接收端文档
│   └── frontend/       # 前端文档
│
├── frontend/           # 前端资源
│   ├── pages/          # 页面模板
│   │   ├── display.html    # 发送端页面
│   │   └── receiver.html   # 接收端页面
│   ├── components/     # 可复用组件
│   └── static/         # 静态资源
│       ├── js/         # JavaScript文件
│       ├── css/        # 样式文件
│       └── models/     # 3D模型文件
│
├── envs/               # 环境配置
│   ├── setup.bat       # Windows环境配置
│   ├── setup.sh        # Linux环境配置
│   ├── setup.py        # Python环境配置
│   ├── meet.yaml       # 通用环境配置
│   ├── meet_windows.yaml  # Windows专用配置
│   ├── meet_ubuntu.yaml   # Ubuntu专用配置
│   └── meet_linux.yaml    # Linux专用配置
│
└── tests/              # 测试用例
    ├── __init__.py
    ├── conftest.py     # 测试配置
    ├── test_system.py  # 系统测试
    ├── test_connect.py # 连接测试
    ├── test_smoother.py # 平滑器测试
    └── connect/        # 连接模块测试
        ├── __init__.py
        ├── test_integration.py    # 集成测试
        └── test_pose_protocol.py  # 协议测试
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

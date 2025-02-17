# Meeting Scene Saver
基于姿态捕获的超低带宽视频会议系统。通过实时捕获人体姿态并在远端重建,实现卫星级别带宽(约10kb/s)下的流畅视频会议。

现有的视频会议都是采用视频传输（连虚拟背景也是），但我们真的有必要每时每刻传输每个像素点吗？如果是4K60帧，每秒数据量有1.49GB，对于精美的电影当然越精细越好，但视频会议并不需要看清楚你桌上的每一点细节，相反，而只需要人的表情和肢体动作，其余的信息反而会成为干扰。

因此我们的项目实现了原理的革新，预先传输模型，在视频通话中只传输捕捉到的人体姿态数据，大大降低了带宽需求。

在2D环境下，带宽降低或许并不必要（当然，我们降低到了卫星通讯水平，可能在极端环境下有用），但如果是未来的VR,AR通话，必然不可能实时传输整个3D环境。
我们的预传模型+实时姿态再渲染的技术在平面视频会议时代或许并无太大优势，但却是未来必然的选择和发展方向

## 核心逻辑
  使用mediapipe模型从摄像头的实时视频流解析姿态关键点坐标，通过网络传输，接收端使用预先传输的初始帧根据关键点坐标进行变形，平滑化后输出视频画面
## 关键性能
- 超低带宽传输: 仅传输姿态数据(~1-10kb/s)
- 实时性能: 端到端延迟<200ms
- 精细姿态:
  - 面部表情(468点)
  - 人体姿态(33点)
  - 手部动作(21点/手)
## 优势功能
- 验证身份保证安全，防止AI换脸
   
    本项目采用的技术容易被用于AI换脸，为保证通话安全，防范电信诈骗，在上传初始参考帧时强制要求人脸识别校验，确认是本人才允许通话
- 人物自动居中，不用调整摄像头角度：
    
    由于使用了姿态数据，可以方便地计算中心点并把人物移动到画面中心，免去了调整摄像头角度的麻烦（尤其是某些摄像头在键盘区的笔记本）
- 方便的裸眼3D：

    姿态关键点数据直接省去了深度估计步骤，使用[Deep3D](https://github.com/HypoX64/Deep3D)项目代码进行简化后方便地达成了裸眼3D效果，接收端资源占用大大减少
  ### 未来扩展
  完成人体建模和骨骼动画部分后可以方便地升级成为VR全息会议

## 快速开始
![alt text](<Screenshot 2025-02-17 165926.png>)
### 环境配置

1. 解压环境：
先安装anaconda https://www.anaconda.com/download/success 下载

打开conda的命令行环境，在项目根目录运行：
  
    conda create -f envs/meet.yaml
运行代码时解释器选择meet环境

    cd meet

运行run.py
程序运行后，打开浏览器访问 `http://127.0.0.1:5000/`。

## 项目结构

```
project/
├── run.py                     # 主程序入口
├── requirements.txt           # Python依赖
├── config/                    # 配置文件
│   ├── __init__.py
│   ├── settings.py           # 全局配置
│   └── config.yaml           # YAML配置
├── camera/                    # 摄像头模块
│   ├── __init__.py
│   └── manager.py            # 摄像头管理
├── pose/                      # 姿态检测模块
│   ├── __init__.py
│   ├── detector.py           # 检测器
│   ├── drawer.py             # 绘制器
│   ├── smoother.py           # 平滑器
│   ├── binding.py            # 绑定器
│   ├── deformer.py           # 变形器
│   ├── types.py              # 类型定义
│   ├── pose_data.py          # 数据结构
│   └── pose_binding.py       # 绑定实现
├── connect/                   # 连接模块
│   ├── __init__.py
│   ├── socket_manager.py     # Socket管理
│   ├── pose_sender.py        # 数据发送
│   ├── pose_protocol.py      # 数据协议
│   └── room_manager.py       # 房间管理
├── receive/                   # 接收端
│   ├── __init__.py
│   ├── app.py                # 应用程序
│   ├── manager.py            # 管理器
│   └── static.py             # 静态资源
├── audio/                     # 音频
│   ├── __init__.py
│   └── processor.py          # 处理器
├── tools/                     # 工具集
│   ├── __init__.py
│   ├── demo.py               # 演示器
│   └── to_center.py          # 居中器
├── docs/                      # 文档
│   ├── README.md             # 索引
│   ├── config/               # 配置说明
│   ├── testing/              # 测试说明
│   ├── usage/                # 使用说明
│   ├── pose/                 # 姿态说明
│   ├── room/                 # 房间说明
│   ├── receive/              # 接收说明
│   └── frontend/             # 前端说明
├── frontend/                  # 前端
│   ├── pages/                # 页面
│   │   ├── display.html      # 发送页
│   │   └── receiver.html     # 接收页
│   ├── components/           # 组件
│   └── static/               # 资源
│       ├── js/               # 脚本
│       ├── css/              # 样式
│       └── models/           # 模型
├── envs/                     # 环境
│   ├── setup.bat             # Windows配置
│   ├── setup.sh              # Linux配置
│   ├── setup.py             # Python配置
│   └── meet.yaml            # 通用配置
└── tests/                    # 测试
    ├── __init__.py
    ├── conftest.py          # 配置
    ├── test_system.py       # 系统测试
    ├── test_connect.py      # 连接测试
    ├── test_smoother.py     # 平滑测试
    ├── test_to_center.py    # 居中测试
    └── connect/             # 连接测试
        ├── __init__.py
        ├── test_integration.py    # 集成
        └── test_pose_protocol.py  # 协议
```

## 项目规模
```
类别统计 (2024.01)
-------------------------------------------------------------------------------
语言类型                     文件数          空行          注释          代码行
-------------------------------------------------------------------------------
Python                        156           4632          4249         18200
Markdown                       50           1218             9          5937
HTML                           4            232             8          1514
JavaScript                     2             84            30           482
配置文件                       7             43            35           394
其他                          8             59            15            288
-------------------------------------------------------------------------------
总计                         227           6268          4346         26815
-------------------------------------------------------------------------------
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
````

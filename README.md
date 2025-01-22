# Meeting Scene Saver
基于姿态捕获的超低带宽视频会议系统。通过实时捕获人体姿态并在远端重建,实现卫星级别带宽(约10kb/s)下的流畅视频会议。

现有的视频会议都是采用视频传输（连虚拟背景也是），我们的项目实现了原理的革新，预先传输模型，在视频通话中只传输捕捉到的人体姿态数据，大大降低了带宽需求。

## 项目简介
在正式的视频会议中，你是否曾因杂乱的房间和难以调整的摄像头角度或自己不佳的仪容仪表而烦恼，或是被突然出现的异物打断重要的采访，亦或是因网络质量不佳而导致会议卡顿？

但是仔细想来，视频会议并没有必要拍摄你所处环境中所有（可能让你感到尴尬）的东西；事实上，只需要你的声音与动作。那么，为什么不提前录入你的身体模型，这样我们就只需要传输捕获的关节数据了，而且从根本上避免了unwanted objects的闯入--我们只传输必要的数据（你的动作），大大减少了数据传输压力，让你在网络欠佳是仍能顺畅通话。另外，与传统的虚拟背景不同，你仍然可以"真实"地出现在会议中--背景可以实地拍摄，模型也可以真实录入，我们只是让你以你最好的面貌出现在会议中,无论是弱网下稳定的通话质量还是更加稳定的虚拟背景（即使系统卡顿也不会显示原画面的固有安全），都会给你满满的安全感，让你的会议更加顺畅自信！

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
=======
由于原理的创新，即使我们的系统出现故障，也不会显现出你未经收拾的房间的画面，而只会显示背景，大大提高了不收拾房间就开正式会议的危险程度（我一直害怕传统的虚拟背景会突然崩溃）；更好的是，我们正在努力通过得到的数据自动把你摆到正确的位置上--再也不用调整摄像头角度了，尤其是如果你的摄像头在笔记本电脑的键盘上，你就会知道这是多么有用。

本项目基于 MediaPipe 的实时人体姿态估计与异物消除系统，实现了上述功能。

希望我们的项目能让你轻松而放心地以最佳的真实面貌地出现在正式的视频会议里。
>>>>>>> fdf218d3e97c4fe6232fb8e4e12b32bd4f93348a

## 快速开始

### 环境配置

1. 安装 Anaconda
   - 从 https://www.anaconda.com/download 下载并安装

2. 创建虚拟环境
   ```bash
   # 创建环境
   conda env create -f envs/meet1.yaml
   ```
如果失败，可以使用对应平台的配置文件：
- Windows: `envs/meet_windows.yaml`
- Linux: `envs/meet_linux.yaml`

3. 激活环境
   ```bash
   conda activate meet
   ```

<<<<<<< HEAD
      (若是windows环境)：mkdir meet6

      (若linux或wsl) mkdir -p meet6
=======
注意：环境配置文件适用于所有平台(Windows/Linux/MacOS)。如果在特定平台遇到问题，请参考 [常见问题](docs/troubleshooting.md)。
>>>>>>> fdf218d3e97c4fe6232fb8e4e12b32bd4f93348a

### 运行程序

1. 启动发送端
   ```bash
   python run.py send
   ```

<<<<<<< HEAD
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

=======
2. 访问页面
   打开浏览器访问 `http://127.0.0.1:5000/`

## 功能

*   **实时人体姿态估计**: 使用 MediaPipe 实时、准确地捕捉人体姿态。
*   **智能异物消除**: 通过模型提取和重建，自动去除背景和干扰人体姿态识别的异物。
*   **背景替换**: 支持上传自定义背景图片，并将处理后的姿态模型渲染到新的背景上。
*   **摄像头/视频流支持**: 支持摄像头实时捕捉和视频文件导入。
*   **用户友好界面**: 提供直观的界面，方便用户操作和查看结果。

## 技术方案

*   **姿态估计**: MediaPipe Pose
*   **图像处理**: OpenCV
*   **3D 模型渲染**: Three.js (或其他合适的 3D 渲染库)
*   **后端**: Flask

## 项目结构
```
project/
├── pose/                   # 姿态处理模块
│   ├── detector.py        # 姿态检测器
│   ├── drawer.py          # 姿态绘制
│   └── types.py           # 数据类型定义
│
├── receive/               # 接收端模块
│   ├── app.py            # 接收端应用
│   ├── manager.py        # 接收端管理器
│   ├── processor.py      # 数据处理器
│   └── display.py        # 显示管理器
│
├── connect/               # 网络连接模块
│   ├── socket_manager.py  # Socket管理
│   └── pose_sender.py     # 姿态数据发送
│
├── tools/                # 工具脚本
│   ├── demo.py           # 演示程序
│   └── create_model.py   # 模型创建工具
│
├── utils/                # 工具函数
│   ├── logger.py         # 日志工具
│   ├── image.py          # 图像处理
│   └── compression.py    # 数据压缩
│
├── frontend/             # 前端界面
│   ├── pages/           # 页面模板
│   │   ├── display.html    # 发送端页面
│   │   └── receiver.html   # 接收端页面
│   ├── static/          # 静态资源
│   │   ├── js/         # JavaScript文件
│   │   ├── css/        # 样式文件
│   │   └── img/        # 图片资源
│   └── components/      # 可复用组件
│       └── room_controls.html  # 房间控制组件
│
├── config/              # 配置文件
│   └── settings.py      # 全局配置
│
├── tests/               # 测试用例
│   ├── conftest.py     # 测试配置
│   ├── connect/        # 连接模块测试
│   ├── pose/           # 姿态模块测试
│   ├── integration/    # 集成测试
│   └── performance/    # 性能测试
│
├── docs/                # 文档
│   ├── pose/           # 姿态模块文档
│   ├── connect/        # 连接模块文档
│   ├── config/         # 配置说明
│   ├── testing/        # 测试说明
│   └── usage/          # 使用说明
│
├── run.py              # 主程序入口
├── pytest.ini          # 测试配置
└── envs/               # 环境配置
     ├── meet_windows.yaml  # Windows环境配置
     └── meet_linux.yaml    # Linux环境配置
```

## 主要模块说明

### 姿态处理 (pose/)
- detector.py: MediaPipe姿态检测
- drawer.py: 姿态可视化绘制
- types.py: 数据结构定义

### 网络连接 (connect/)
- socket_manager.py: WebSocket连接管理
- pose_sender.py: 姿态数据发送

### 前端界面 (frontend/)
- display.html: 发送端页面
- receiver.html: 接收端页面
- room_controls.html: 房间控制组件

### 测试用例 (tests/)
- 单元测试
- 集成测试
- 性能测试
- 稳定性测试

## 启动说明

### 发送端
```bash
python run.py send [--camera CAMERA_ID] [--room ROOM_ID]
```

### 接收端
```bash
python run.py receive [--room ROOM_ID]
```

### 启动演示程序
```bash
python run.py demo
```

## 未来改进

*   优化算法，提高处理速度和准确性。
*   支持更多类型的异物消除。
*   增加对不同分辨率和帧率的支持。
*   开发移动端应用。

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件 (您需要创建 LICENSE 文件并添加 MIT 许可证内容)。
>>>>>>> fdf218d3e97c4fe6232fb8e4e12b32bd4f93348a

## 安全功能

### 认证
系统使用 JWT (JSON Web Token) 进行认证。在建立连接前需要先进行认证：

```python
# 认证示例
socket_manager = SocketManager(socketio, audio_processor)
socket_manager.authenticate({
    'username': 'your_username',
    'password': 'your_password'
})
```

### 数据安全
- 数据压缩：自动压缩大于1KB的数据
- 数据签名：使用JWT对数据进行签名，确保完整性
- 时间戳：防止重放攻击

### 配置
可以在 config.yaml 中配置安全选项：

```yaml
socket:
  security:
    encryption_enabled: true
    compression_level: 6
    token_expiry: 3600
```

## 环境要求

Linux用户请使用 `conda env create -f meet_ubuntu.yaml` 创建环境，支持Ubuntu 22.04及以上版本。

<<<<<<< HEAD
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
=======
## 安装步骤
1. 安装Anaconda或Miniconda
2. 创建虚拟环境
...
>>>>>>> fdf218d3e97c4fe6232fb8e4e12b32bd4f93348a

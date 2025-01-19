# Meeting Scene Saver: 基于 MediaPipe 的实时人体姿态估计与异物消除系统

## 项目简介
在正式的视频会议中，你是否曾因杂乱的房间和难以调整的摄像头角度或自己不佳的仪容仪表而烦恼，或是被突然出现的异物打断重要的采访，亦或是因网络质量不佳而导致会议卡顿？

但是仔细想来，视频会议并没有必要拍摄你所处环境中所有（可能让你感到尴尬）的东西；事实上，只需要你的声音与动作。那么，为什么不提前录入你的身体模型，这样我们就只需要传输捕获的关节数据了，而且从根本上避免了unwanted objects的闯入--我们只传输必要的数据（你的动作），大大减少了数据传输压力，让你在网络欠佳是仍能顺畅通话。另外，与传统的虚拟背景不同，你仍然可以"真实"地出现在会议中--背景可以实地拍摄，模型也可以真实录入，我们只是让你以你最好的面貌出现在会议中。

由于原理的创新，即使我们的系统出现故障，也不会显现出你未经收拾的房间的画面，而只会显示背景，大大提高了不收拾房间就开正式会议的危险程度（我一直害怕传统的虚拟背景会突然崩溃）；更好的是，我们正在努力通过得到的数据自动把你摆到正确的位置上--再也不用调整摄像头角度了，尤其是如果你的摄像头在笔记本电脑的键盘上，你就会知道这是多么有用。

本项目基于 MediaPipe 的实时人体姿态估计与异物消除系统，实现了上述功能。

希望我们的项目能让你轻松而放心地以最佳的真实面貌地出现在正式的视频会议里。

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

注意：环境配置文件适用于所有平台(Windows/Linux/MacOS)。如果在特定平台遇到问题，请参考 [常见问题](docs/troubleshooting.md)。

### 运行程序

1. 启动发送端
   ```bash
   python run.py send
   ```

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

## 安装步骤
1. 安装Anaconda或Miniconda
2. 创建虚拟环境
...

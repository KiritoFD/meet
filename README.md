# Meeting Scene Saver: 基于 MediaPipe 的实时人体姿态估计与异物消除系统

## 项目简介
在正式的视频会议中，你是否曾因杂乱的房间和难以调整的摄像头角度或自己不佳的仪容仪表而烦恼，或是被突然出现的异物打断重要的采访，亦或是因网络质量不佳而导致会议卡顿？

但是仔细想来，视频会议并没有必要拍摄你所处环境中所有（可能让你感到尴尬）的东西；事实上，只需要你的声音与动作。那么，为什么不提前录入你的身体模型，这样我们就只需要传输捕获的关节数据了，而且从根本上避免了unwanted objects的闯入--我们只传输必要的数据（你的动作），大大减少了数据传输压力，让你在网络欠佳是仍能顺畅通话。另外，与传统的虚拟背景不同，你仍然可以"真实"地出现在会议中--背景可以实地拍摄，模型也可以真实录入，我们只是让你以你最好的面貌出现在会议中。

由于原理的创新，即使我们的系统出现故障，也不会显现出你未经收拾的房间的画面，而只会显示背景，大大提高了不收拾房间就开正式会议的危险程度（我一直害怕传统的虚拟背景会突然崩溃）；更好的是，我们正在努力通过得到的数据自动把你摆到正确的位置上--再也不用调整摄像头角度了，尤其是如果你的摄像头在笔记本电脑的键盘上，你就会知道这是多么有用。


## 快速开始

### 方法0：如果你不想自己安装conda,请运行setup.bat（windows）或setup.sh（linux）


如果想自己动手（我们的安装脚本并不那么可靠）

先安装anaconda
https://www.anaconda.com/download/success 下载

### 方法1
    
在项目根目录下运行（根据系统选择对应的配置文件）：

    # Windows系统：
    conda env create -f meet_windows.yaml
    # Linux系统：
    conda env create -f meet_linux.yaml

然后用anaconda prompt 输入以下命令：

    conda activate meet6
之后就可以选择meet6环境运行代码了(选择解释器)


### 方法2：使用打包好的环境

1. 解压环境：

0.安装软件：

先安装anaconda
https://www.anaconda.com/download/success 下载
安装WinRAR用于解压
https://www.win-rar.com/download.html?&L=0
1. 解压环境：

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
meet/
├── frontend/           # 前端资源统一目录
│   ├── static/        # 静态资源
│   │   ├── js/       # JavaScript文件
│   │   │   ├── poseRenderer.js  # 姿态渲染器
│   │   │   └── app.js           # 主应用逻辑
│   │   ├── css/      # 样式文件
│   │   └── img/      # 图片资源
│   └── pages/        # 页面文件
│       ├── components/  # 可复用组件
│       │   └── room_controls.html  # 房间控制组件
│       ├── display.html    # 发送端页面
│       └── receiver.html   # 接收端页面
├── pose/              # 姿态处理模块
│   ├── detector.py    # 姿态检测器
│   ├── processor.py   # 姿态数据处理
│   └── drawer.py      # 姿态绘制
├── room/              # 房间管理模块
│   └── manager.py     # 房间管理器
├── receive/           # 接收端模块
│   ├── app.py        # 接收端应用
│   ├── manager.py    # 接收端管理器
│   ├── transform.py  # 姿态变换
│   └── static.py     # 静态文件服务
├── connect/          # 连接处理模块
│   ├── pose_sender.py    # 姿态数据发送
│   └── socket_manager.py # Socket连接管理
├── utils/            # 工具函数
│   ├── compression.py  # 数据压缩
│   └── image.py       # 图像处理
├── config/           # 配置文件
│   └── settings.py   # 全局配置
├── run.py            # 发送端入口
├── receiver.py       # 接收端入口
├── requirements.txt  # 项目依赖
├── README.md         # 项目说明
└── LICENSE           # 许可证
```

## 未来改进

*   优化算法，提高处理速度和准确性。
*   支持更多类型的异物消除。
*   增加对不同分辨率和帧率的支持。
*   开发移动端应用。

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件 (您需要创建 LICENSE 文件并添加 MIT 许可证内容)。

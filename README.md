# Meeting Scene Saver: 基于 MediaPipe 的实时人体姿态估计与异物消除系统

## 项目简介
现有的视频会议都是采用视频传输（连虚拟背景也是），我们的项目实现了原理的革新，预先传输模型，在视频通话中只传输捕捉到的人体姿态数据，大大降低了带宽需求，同时从根本上避免了异物出现在画面中，还实现了更加流畅自定义的背景。

在2D环境下，带宽降低或许并不必要（当然，我们降低到了卫星通讯水平，可能在极端环境下有用），但如果是未来的VR,AR通话，必然不可能实时传输整个3D环境。
我们的预传模型+实时姿态再渲染的技术在平面视频会议时代或许并无太大优势，但却是未来必然的选择和发展方向。

## 快速开始

### 方法1：使用打包好的环境（推荐）

1. 解压环境：
先安装anaconda https://www.anaconda.com/download/success 下载

win+r 打开cmd 

    cd anaconda3\envs

    mkdir meet6

    cd meet6
复制当前路径

用anaconda prompt 进入项目根目录，输入以下命令：

    tar -xvf meet6.tar.gz -C 
在这一行后粘贴刚才复制的路径，enter

    conda activate meet6
运行代码时解释器选择meet6环境
### 方法2：使用 Docker（可选）

1. 构建并运行：
```bash
docker build -t meet .
docker run -p 5000:5000 meet
```

## 环境要求

- Python 3.12.7
- 主要依赖包：
  - Flask 3.1.0
  - MediaPipe 0.10.20
  - OpenCV 4.10.0.84
  - NumPy 1.26.4

## 功能

*   **实时人体姿态估计**: 使用 MediaPipe 实时、准确地捕捉人体姿态。
*   **智能异物消除**: 通过模型提取和重建，自动去除背景和干扰人体姿态识别的异物。
*   **背景替换**: 支持上传自定义背景图片，并将处理后的姿态模型渲染到新的背景上。
*   **摄像头/视频流支持**: 支持摄像头实时捕捉和视频文件导入。

## 技术方案

*   **姿态估计**: MediaPipe Pose
*   **图像处理**: OpenCV
*   **3D 模型渲染**: Three.js (或其他合适的 3D 渲染库)
*   **后端**: Flask

## 目录说明

*   `capture.py`: 处理摄像头/视频帧，进行姿态估计。
*   `run.py`: 程序入口文件。
*   `src/`: 源代码目录。
    *   `core/`: 核心业务逻辑。
        *   `__init__.py`: 初始化文件。
        *   `video_processor.py`: 视频处理和姿态估计的核心逻辑。
    *   `server.py`: Flask 应用，处理 HTTP 请求和响应。
    *   `static/`: 前端静态资源 (如果存在)。
    *   `templates/`: HTML 模板。
        *   `display.html`: 前端页面，包含 3D 模型渲染和用户交互逻辑。
    *   `utils/`: 工具函数。
        *   `__init__.py`: 初始化文件。
        *   `logger.py`: 日志配置。
*   `templates/`: HTML 模板。
    *   `display.html`: 前端页面，包含 3D 模型渲染和用户交互逻辑。

## 未来改进

*   优化算法，提高处理速度和准确性。
*   支持更多类型的异物消除。
*   增加对不同分辨率和帧率的支持。
*   开发移动端应用。

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件 (您需要创建 LICENSE 文件并添加 MIT 许可证内容)。


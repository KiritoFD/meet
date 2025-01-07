# Meeting Scene Saver

基于姿态识别的会议场景保存系统。支持全身姿态、手部动作和面部表情的实时识别。

## 环境配置

### 1. 创建并激活 conda 环境

```bash
# 创建环境
conda create -n meet python=3.8 -y

# 激活环境
conda activate meet

# 安装基础依赖
conda install -c conda-forge opencv numpy flask -y

# 安装其他依赖
pip install mediapipe werkzeug
```

### 2. 项目结构

```
new/
├── run.py              # 主程序入口
├── requirements.txt    # 依赖列表
├── static/            # 静态资源
│   ├── css/          # 样式文件
│   ├── js/           # JavaScript文件
│   ├── models/       # 3D模型文件
│   └── backgrounds/  # 背景图片
├── templates/         # 模板文件
│   └── index.html    # 主页面
└── uploads/          # 上传文件存储
```

### 3. 运行项目

```bash
# 确保在meet环境中
conda activate meet

# 运行服务器
python run.py
```

访问 http://localhost:5000 查看应用。

## 功能特性

- 实时视频捕获和显示
- 多模态识别
  - 全身姿态检测
  - 手部动作识别 (支持双手)
  - 面部网格检测 (468个关键点)
  - 眼睛和虹膜跟踪
- 关键点标记和连线
- 姿态数据记录
- 截图功能
- 状态监控

## 开发说明

### 1. 代码结构

- `run.py`: 主服务器程序
- `static/js/app.js`: 前端交互逻辑
- `static/css/style.css`: 页面样式
- `templates/index.html`: 页面模板

### 2. 主要依赖

- Flask: Web服务器框架
- OpenCV: 视频处理
- MediaPipe: 多模态识别
  - Pose: 姿态检测
  - Hands: 手部识别
  - FaceMesh: 面部网格
- NumPy: 数据处理

### 3. 开发模式

```bash
# 启动开发服务器
python run.py
```

服务器将以调试模式运行,支持代码热重载。

## 注意事项

1. 确保摄像头可用且未被其他程序占用
2. 保持良好的光照条件以提高识别准确度
3. 建议在稳定的网络环境下使用
4. 面部识别需要较近的距离
5. 手部识别需要手部在画面中清晰可见

## 常见问题

1. 摄像头无法启动
   - 检查摄像头是否被其他程序占用
   - 确认摄像头驱动正确安装

2. 识别不准确
   - 调整光照条件
   - 确保目标在画面中清晰可见
   - 避免复杂背景干扰
   - 保持适当距离(面部识别约50cm最佳)

## 更新日志

### v1.1.0
- 添加手部动作识别
- 添加面部网格检测
- 添加眼睛和虹膜跟踪
- 优化数据传输格式

### v1.0.0
- 基础功能实现
- 姿态检测和显示
- 实时数据获取
- 截图功能 
# 项目文件结构

meet/
├── run.py                     # 主程序入口，启动服务器和初始化系统
├── requirements.txt           # Python依赖包清单
├── README.md                  # 项目说明文档
│
├── config/                    # 配置模块：系统全局配置
│   ├── __init__.py           # 配置模块初始化
│   ├── settings.py           # 全局配置参数定义
│   ├── config.yaml           # 可配置参数YAML文件
│   ├── jitsi_config.py       # Jitsi配置
│   └── security_config.py    # 安全配置
│
├── camera/                    # 摄像头模块：视频采集和预处理
│   ├── __init__.py           # 模块初始化
│   ├── manager.py            # 摄像头管理：初始化、采集、控制
│   ├── calibration.py        # 摄像头校准：畸变校正
│   ├── filters.py            # 图像滤镜：预处理、增强
│   └── exceptions.py         # 异常定义
│
├── pose/                      # 姿态检测模块：核心算法实现
│   ├── __init__.py           # 模块初始化
│   ├── detector.py           # 姿态检测器：MediaPipe实现
│   ├── drawer.py             # 姿态绘制：关键点可视化
│   ├── smoother.py           # 平滑处理：卡尔曼滤波
│   ├── binding.py            # 骨骼绑定：权重计算
│   ├── deformer.py           # 网格变形：姿态驱动
│   ├── types.py              # 类型定义：数据结构
│   ├── pose_data.py          # 姿态数据：数据封装
│   ├── pose_binding.py       # 绑定实现：骨骼绑定逻辑
│   ├── utils/                # 工具函数
│   │   ├── __init__.py
│   │   ├── math.py          # 数学计算
│   │   └── visualization.py  # 可视化
│   └── tests/                # 单元测试
│       ├── __init__.py
│       └── test_detector.py  # 检测器测试
│
├── connect/                   # 连接模块：网络传输实现
│   ├── __init__.py           # 模块初始化
│   ├── socket_manager.py     # Socket管理：WebSocket服务
│   ├── pose_sender.py        # 姿态发送：数据传输
│   ├── pose_protocol.py      # 数据协议：编解码规范
│   ├── room_manager.py       # 房间管理：多人会议
│   ├── errors.py             # 错误定义
│   ├── security/             # 安全子模块
│   │   ├── __init__.py
│   │   ├── auth.py          # 身份认证
│   │   └── encryption.py    # 数据加密
│   ├── jitsi/                # Jitsi集成子模块
│   │   ├── __init__.py      # 子模块初始化
│   │   ├── transport.py     # 传输层：Jitsi数据通道
│   │   ├── processor.py     # 处理器：数据处理
│   │   ├── room.py          # 会议室管理
│   │   └── serializer.py    # 数据序列化
│   └── tests/                # 连接测试
│       ├── __init__.py
│       └── test_protocol.py  # 协议测试
│
├── receive/                   # 接收端模块：数据接收和显示
│   ├── __init__.py           # 模块初始化
│   ├── app.py                # 接收应用：Flask服务
│   ├── manager.py            # 接收管理：数据处理
│   ├── renderer.py           # 渲染器：WebGL实现
│   ├── buffer.py             # 缓冲管理
│   ├── static/               # 静态资源
│   │   ├── js/              # JavaScript文件
│   │   │   ├── app.js       # 主程序
│   │   │   ├── webgl.js     # WebGL渲染
│   │   │   └── socket.js    # Socket通信
│   │   └── css/             # 样式表
│   │       └── style.css    # 主样式
│   └── templates/            # 模板文件
│       └── index.html        # 主页面
│
├── audio/                     # 音频模块：音频处理
│   ├── __init__.py           # 模块初始化
│   ├── processor.py          # 音频处理：编解码
│   ├── encoder.py            # 音频编码：压缩
│   ├── stream.py             # 音频流：传输
│   └── effects/              # 音效处理
│       ├── __init__.py
│       └── noise_reduction.py # 降噪
│
├── tools/                     # 工具模块：开发辅助
│   ├── __init__.py           # 模块初始化
│   ├── demo.py               # 演示程序：功能展示
│   ├── profiler.py           # 性能分析：性能监控
│   ├── to_center.py          # 居中处理：画面校正
│   ├── logger.py             # 日志工具：日志记录
│   └── utils/                # 通用工具
│       ├── __init__.py
│       ├── image.py         # 图像处理
│       └── file.py          # 文件操作
│
├── frontend/                  # 前端模块：用户界面
│   ├── pages/                # 页面文件
│   │   ├── display.html      # 发送页面：视频采集
│   │   ├── receiver.html     # 接收页面：视频显示
│   │   └── room.html         # 房间页面：会议室
│   ├── components/           # Vue组件
│   │   ├── Camera.vue        # 摄像头组件
│   │   ├── PoseView.vue      # 姿态显示
│   │   ├── Controls.vue      # 控制面板
│   │   └── RoomList.vue      # 房间列表
│   └── static/               # 静态资源
│       ├── js/               # JavaScript
│       │   ├── app.js        # 主程序
│       │   ├── webgl.js      # WebGL渲染
│       │   └── socket.js     # Socket通信
│       ├── css/              # 样式表
│       │   ├── main.css      # 主样式
│       │   └── components.css # 组件样式
│       └── models/           # 3D模型
│           └── avatar/       # 虚拟形象
│
├── tests/                     # 测试模块：单元测试和集成测试
│   ├── __init__.py           # 模块初始化
│   ├── conftest.py           # 测试配置：pytest配置
│   ├── test_system.py        # 系统测试：端到端测试
│   ├── test_camera.py        # 摄像头测试
│   ├── test_pose.py          # 姿态测试
│   ├── test_connect.py       # 连接测试
│   ├── test_receive.py       # 接收测试
│   ├── test_smoother.py      # 平滑器测试
│   ├── test_to_center.py     # 居中测试
│   └── connect/              # 连接测试目录
│       ├── __init__.py
│       ├── test_integration.py # 集成测试
│       └── test_pose_protocol.py # 协议测试
│
├── docs/                      # 文档模块：项目文档
│   ├── README.md             # 文档索引
│   ├── architecture/         # 架构文档
│   │   ├── overview.md      # 架构概览
│   │   └── components.md    # 组件说明
│   ├── api/                  # API文档
│   │   ├── camera.md        # 摄像头API
│   │   ├── pose.md          # 姿态API
│   │   └── connect.md       # 连接API
│   ├── deployment/           # 部署文档
│   │   ├── setup.md         # 环境搭建
│   │   └── config.md        # 配置说明
│   └── development/          # 开发文档
│       ├── guide.md         # 开发指南
│       └── standards.md     # 开发规范
│
└── envs/                     # 环境模块：环境配置
    ├── setup.bat             # Windows配置脚本
    ├── setup.sh              # Linux配置脚本
    ├── setup.py              # Python环境配置
    ├── requirements.txt      # 依赖清单
    └── meet.yaml             # Conda环境配置 
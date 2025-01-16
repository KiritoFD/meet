# Connect模块架构

## 目录结构
```
connect/
├── __init__.py
├── socket_manager.py    # 基础连接管理
├── room_manager.py      # 房间管理
├── pose_sender.py       # 姿态数据发送
├── utils/
│   ├── __init__.py
│   ├── monitoring.py    # 性能监控
│   └── errors.py        # 错误定义
└── config.py           # 配置定义
```

## 测试结构
```
# 接收端模块

## 模块结构
```
receive/
├── app.py          # 接收端应用程序
├── manager.py      # 接收端管理器
├── processor.py    # 数据处理器
└── display.py      # 显示管理器
```

## 功能说明
1. 接收姿态数据
2. 实时渲染显示
3. 房间管理
4. 状态同步

## 组件说明

### 接收端应用 (app.py)
```python
class ReceiveApp:
    """接收端主应用"""
    def __init__(self, config: Dict = None):
        self.manager = ReceiveManager()
        self.processor = DataProcessor()
        self.display = DisplayManager()
        
    async def start(self):
        """启动接收端"""
        await self.manager.connect()
        await self.processor.init()
        self.display.start()
```

### 接收端管理器 (manager.py)
- 管理Socket连接
- 处理房间加入/退出
- 维护连接状态

### 数据处理器 (processor.py)
- 解压数据
- 姿态重建
- 帧同步

### 显示管理器 (display.py)
- 窗口管理
- 渲染控制
- 性能监控 
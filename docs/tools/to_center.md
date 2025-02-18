# 姿态居中工具 (tools.to_center)

## 功能说明

将人体姿态关键点数据居中到画面中心位置(0.5, 0.5)。该模块主要用于标准化姿态数据的位置。

### 特点
# ...existing content...

## 使用方法

1. 基本使用:
```python
from tools import to_center
from pose.pose_data import PoseData

# pose_data 是你的姿态数据
success = to_center(pose_data)  # 原地修改数据
if success:
    print("居中成功")
else:
    print("居中失败")
```

# ...rest of the existing content...

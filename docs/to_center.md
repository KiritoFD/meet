# 姿态居中模块 (to_center)

## 功能说明

将人体姿态关键点数据居中到画面中心位置(0.5, 0.5)。该模块主要用于标准化姿态数据的位置。

### 特点

- 使用躯干关键点加权计算中心点
- 自动处理不可见和低质量关键点
- 包含防抖动机制
- 保持姿态的相对结构不变
- 所有坐标保持在[0,1]范围内

## 使用方法

1. 基本使用:
```python
from to_center import to_center
from pose.pose_data import PoseData

# pose_data 是你的姿态数据
success = to_center(pose_data)  # 原地修改数据
if success:
    print("居中成功")
else:
    print("居中失败")
```

2. 开启调试日志:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 核心关键点权重

模块使用以下关键点计算中心：

- 左肩膀 (权重: 2.0)
- 右肩膀 (权重: 2.0)
- 左胯 (权重: 1.5)
- 右胯 (权重: 1.5)
- 颈部 (权重: 1.0)

## 示例

完整使用示例：

```python
import logging
from pose.pose_data import PoseData, Landmark
from to_center import to_center

# 设置日志级别以查看详细信息
logging.basicConfig(level=logging.DEBUG)

# 创建测试姿态数据
landmarks = [Landmark(x=0.3, y=0.3, z=0, visibility=1.0) for _ in range(33)]
pose_data = PoseData(
    landmarks=landmarks,
    timestamp=0,
    confidence=1.0
)

# 执行居中操作
success = to_center(pose_data)

# 检查结果
if success:
    print("姿态已成功居中")
    print(f"中心点坐标: ({pose_data.landmarks[1].x:.2f}, {pose_data.landmarks[1].y:.2f})")
else:
    print("居中失败，可能是关键点不足或可见度太低")
```

## 注意事项

1. 函数会原地修改输入的姿态数据
2. 至少需要2个可见的核心关键点才能进行有效居中
3. 关键点可见度阈值为0.7
4. 包含防抖动处理，大幅度变化会被平滑
5. 所有坐标会被限制在[0,1]范围内

## 返回值说明

- `True`: 成功完成居中操作
- `False`: 居中失败（无效数据或关键点不足）

## 调试

设置 DEBUG 日志级别可以看到：
- 当前中心点位置
- 计算的偏移量
- 平滑处理的触发
- 关键点可用性信息

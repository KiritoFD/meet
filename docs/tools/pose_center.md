# 姿态居中工具(Pose Center Tool)

## 功能介绍
姿态居中工具用于将检测到的人体姿态关键点自动调整到画面中心位置。主要特点：
- 加权中心计算，重点关注核心身体部位
- 平滑处理防抖动
- 异常点过滤
- 可配置的参数系统

## 使用方法

### 基础用法
```python
from tools.to_center import to_center
from pose.pose_data import PoseData

# 使用默认配置居中处理
success = to_center(pose_data)
```

### 自定义配置
```python
# 临时覆盖部分配置
custom_config = {
    'smoothing_factor': 0.5,    # 增加平滑程度
    'max_offset': 0.15         # 允许更大的偏移
}
success = to_center(pose_data, custom_config)
```

## 配置参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| visibility_threshold | float | 0.7 | 关键点可见度阈值 |
| min_valid_points | int | 2 | 最小有效关键点数量 |
| smoothing_factor | float | 0.3 | 平滑因子(0-1) |
| max_offset | float | 0.1 | 最大允许偏移距离 |
| outlier_threshold | float | 0.2 | 异常点判定阈值 |

## 核心关键点权重

| 关键点 | 权重 | 说明 |
|--------|------|------|
| left_shoulder | 2.0 | 左肩 |
| right_shoulder | 2.0 | 右肩 |
| left_hip | 1.5 | 左胯 |
| right_hip | 1.5 | 右胯 |
| neck | 1.0 | 颈部 |

## 实现细节

### 居中算法
1. 计算加权中心点
2. 对比目标位置(0.5, 0.5)
3. 计算需要的偏移量
4. 应用平滑处理
5. 限制最大偏移
6. 更新所有关键点坐标

### 异常处理
- 过滤可见度低的点
- 移除离群点
- 平滑大幅度变化
- 确保坐标在[0,1]范围内

## 注意事项
1. 需要足够数量的有效关键点才能准确居中
2. 平滑处理可能导致轻微的延迟
3. 配置参数会影响响应速度和稳定性

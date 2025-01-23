# 姿态绑定器(PoseBinding)

## 1. 功能说明
负责处理姿态关键点到图像区域的绑定，包括区域生成、权重计算和更新。

## 2. 配置说明

### 2.1 绑定配置(BindingConfig)
```python
@dataclass
class BindingConfig:
    smoothing_factor: float      # 平滑因子
    min_confidence: float        # 最小置信度
    joint_limits: Dict[str, Tuple[float, float]]  # 关节角度限制
```

### 2.2 区域配置
```python
region_configs = {
    # 身体区域
    'torso': [11, 12, 23, 24],
    'left_upper_arm': [11, 13],
    'left_lower_arm': [13, 15],
    'right_upper_arm': [12, 14],
    'right_lower_arm': [14, 16],
    'left_upper_leg': [23, 25],
    'left_lower_leg': [25, 27],
    'right_upper_leg': [24, 26],
    'right_lower_leg': [26, 28],
    
    # 面部区域
    'face_contour': [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109],
    'left_eyebrow': [70,63,105,66,107,55,65],
    'right_eyebrow': [336,296,334,293,300,285,295],
    'left_eye': [33,246,161,160,159,158,157,173,133],
    'right_eye': [362,398,384,385,386,387,388,466,263],
    'nose': [168,6,197,195,5,4,1,19,94,2],
    'mouth': [0,267,269,270,409,291,375,321,405,314,17,84,181,91,146,61,185,40,39,37]
}
```

## 3. 主要API

### 3.1 create_binding()
```python
def create_binding(self, frame: np.ndarray, pose_data: PoseData) -> List[DeformRegion]:
    """创建初始帧的区域绑定
    
    Args:
        frame: 初始图像帧
        pose_data: 姿态数据
        
    Returns:
        List[DeformRegion]: 变形区域列表
        
    Raises:
        ValueError: 当输入无效时
    """
```

### 3.2 update_binding()
```python
def update_binding(self, regions: List[DeformRegion], pose: PoseData) -> List[DeformRegion]:
    """更新绑定信息
    
    Args:
        regions: 现有的区域列表
        pose: 新的姿态数据
        
    Returns:
        List[DeformRegion]: 更新后的区域列表
    """
```

## 4. 数据结构

### 4.1 DeformRegion
```python
@dataclass
class DeformRegion:
    center: np.ndarray          # 区域中心点坐标 [x, y]
    binding_points: List[BindingPoint]  # 绑定点列表
    mask: Optional[np.ndarray]  # 区域蒙版 (H, W), uint8
```

### 4.2 BindingPoint
```python
@dataclass
class BindingPoint:
    landmark_index: int        # 对应的关键点索引
    local_coords: np.ndarray   # 相对于中心点的局部坐标 [dx, dy]
    weight: float             # 影响权重 (0-1)
```

## 5. 技术要求

### 5.1 区域生成
- 必需区域：
  1. 躯干区域：
     - torso: 躯干区域(关键点11,12,23,24)

  2. 手臂区域：
     - left_upper_arm: 左上臂(关键点11,13)
     - left_lower_arm: 左下臂(关键点13,15)
     - right_upper_arm: 右上臂(关键点12,14)
     - right_lower_arm: 右下臂(关键点14,16)

  3. 腿部区域：
     - left_upper_leg: 左大腿(关键点23,25)
     - left_lower_leg: 左小腿(关键点25,27)
     - right_upper_leg: 右大腿(关键点24,26)
     - right_lower_leg: 右小腿(关键点26,28)

  4. 面部区域：
     - face_contour: 面部轮廓(关键点10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109)
     - left_eyebrow: 左眉(关键点70,63,105,66,107,55,65)
     - right_eyebrow: 右眉(关键点336,296,334,293,300,285,295)
     - left_eye: 左眼(关键点33,246,161,160,159,158,157,173,133)
     - right_eye: 右眼(关键点362,398,384,385,386,387,388,466,263)
     - nose: 鼻子(关键点168,6,197,195,5,4,1,19,94,2)
     - mouth: 嘴部(关键点0,267,269,270,409,291,375,321,405,314,17,84,181,91,146,61,185,40,39,37)

### 5.2 权重计算
- 躯干区域：
  - 关节点权重：0.6
  - 端点权重：0.4
- 肢体区域：
  - 端点权重：0.7
  - 控制点权重：0.3
- 面部区域：
  - 轮廓点权重：0.5
  - 特征点权重：0.8
  - 控制点权重：0.3

### 5.3 蒙版要求
- 数据类型：uint8
- 尺寸：与输入图像相同
- 值范围：0-255
- 边缘平滑：高斯模糊(21x21)

### 5.4 性能要求
- 绑定创建：< 10ms
- 绑定更新：< 5ms
- 内存使用：< 100MB

## 6. 错误处理
- 输入验证
- 关键点可见度检查
- 缓存机制
- 优雅降级

## 7. 注意事项
1. 所有坐标需转换到图像空间
2. 权重需要归一化
3. 保持区域连续性
4. 处理边界情况

# 姿态绑定器(PoseBinder)

## 功能说明
负责初始帧的区域绑定和参考姿态设置。

## 绑定配置
```python
BIND_CONFIG = {
    'min_region_size': 20,    # 最小区域大小
    'max_regions': 20,        # 最大区域数量
    'overlap_threshold': 0.3  # 重叠阈值
}
```

## API说明

### create_binding()
```python
def create_binding(self, frame: np.ndarray, pose_data: PoseData) -> List[DeformRegion]:
    """创建初始帧的区域绑定
    
    Args:
        frame: 初始图像帧
        pose_data: 姿态数据
        
    Returns:
        List[DeformRegion]: 变形区域列表
    """
```

### 区域生成
```python
def generate_regions(self, pose_data: PoseData) -> List[DeformRegion]:
    """生成变形区域
    
    Args:
        pose_data: 姿态数据
        
    Returns:
        List[DeformRegion]: 变形区域列表
    """
    regions = []
    
    # 生成躯干区域
    torso = self._create_torso_region(pose_data)
    regions.append(torso)
    
    # 生成四肢区域
    limbs = self._create_limb_regions(pose_data)
    regions.extend(limbs)
    
    # 生成头部区域
    head = self._create_head_region(pose_data)
    regions.append(head)
    
    return regions
```

## 区域定义
1. 躯干区域
   - 肩部关键点
   - 臀部关键点
   - 躯干中心线

2. 四肢区域
   - 上臂和前臂
   - 大腿和小腿
   - 关节连接

3. 头部区域
   - 面部特征点
   - 颈部连接
   - 头部轮廓

## 1. 输出数据结构

输出 Dict[str, DeformRegion]，包含以下区域:

### 1.1 必需区域
- torso: 躯干区域(关键点11,12,23,24)
- left_upper_arm: 左上臂(关键点11,13)
- left_lower_arm: 左下臂(关键点13,15)
- right_upper_arm: 右上臂(关键点12,14)
- right_lower_arm: 右下臂(关键点14,16)
- left_upper_leg: 左大腿(关键点23,25)
- left_lower_leg: 左小腿(关键点25,27)
- right_upper_leg: 右大腿(关键点24,26)
- right_lower_leg: 右小腿(关键点26,28)

### 1.2 DeformRegion 结构
```python
@dataclass
class DeformRegion:
    center: np.ndarray          # 区域中心点坐标 [x, y]
    binding_points: List[BindingPoint]  # 绑定点列表
    mask: np.ndarray           # 区域蒙版 (H, W), uint8
```

### 1.3 BindingPoint 结构
```python
@dataclass
class BindingPoint:
    landmark_index: int        # 对应的关键点索引
    local_coords: np.ndarray   # 相对于中心点的局部坐标 [dx, dy]
    weight: float             # 影响权重 (0-1)
```

## 2. 技术要求

### 2.1 蒙版要求
- 数据类型: uint8
- 尺寸: 与输入图像相同
- 值范围: 0-255
- 边缘需平滑过渡(高斯模糊)
- 蒙版边缘宽度: 20像素

### 2.2 绑定点要求
- 每个区域至少3个点(用于仿射变换)
- 区域内权重和为1.0
- 关节点权重0.6-0.7
- 端点权重0.3-0.4
- 不足3点时，添加半径50像素的控制点

### 2.3 区域重叠
- 相邻区域20-30%重叠
- 重叠区域权重平滑过渡
- 使用高斯模糊进行边缘平滑

## 3. 性能要求

### 3.1 时间性能
- 初始绑定处理 < 10ms

### 3.2 资源使用
- 静态内存 < 50MB
- 避免频繁内存分配
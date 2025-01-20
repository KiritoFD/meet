# 姿态变形器(PoseDeformer)

## 功能说明
负责根据姿态变化对图像进行变形处理。

## 变形配置
```python
DEFORM_CONFIG = {
    'smoothing_window': 5,     # 平滑窗口大小
    'smoothing_factor': 0.3,   # 平滑系数
    'blend_radius': 20,        # 混合半径
    'min_scale': 0.5,         # 最小缩放
    'max_scale': 2.0          # 最大缩放
}
```

## API说明

### deform_frame()
```python
def deform_frame(self, frame, regions: List[DeformRegion], pose_data: PoseData) -> np.ndarray:
    """对图像进行变形处理
    
    Args:
        frame: 输入图像帧
        regions: 变形区域列表
        pose_data: 姿态数据
        
    Returns:
        np.ndarray: 变形后的图像
    """
```

### 变形区域定义
```python
@dataclass
class DeformRegion:
    """变形区域定义"""
    keypoints: List[int]      # 关键点ID列表
    mask: np.ndarray         # 区域掩码
    reference_pose: PoseData  # 参考姿态
```

## 变形算法
1. 区域划分
   - 根据关键点划分区域
   - 生成区域掩码
   - 计算区域权重

2. 姿态变换
   - 计算仿射变换
   - 应用变形矩阵
   - 区域混合

3. 图像处理
   - 双线性插值
   - 边缘平滑
   - 颜色校正

## 变形约束
1. 区域变形
   - 最小缩放: 0.5x
   - 最大缩放: 2.0x
   - 保持区域连续性

2. 时间平滑
   - 平滑窗口: 5帧
   - 平滑因子: 0.3
   - 避免抖动

3. 区域混合
   - 混合半径: 20像素
   - 高斯模糊过渡
   - 无缝拼接

## 性能说明
- 单帧处理 < 10ms
- 内存占用 < 100MB
- 支持30fps实时处理 

# 姿态变形器 (PoseDeformer) 实现原理

## 核心思路

PoseDeformer 通过以下步骤将输入图像根据目标姿态进行变形：

1. 全局变形：计算整体的旋转、缩放和平移
2. 区域变形：对不同身体部位进行局部变形
3. 区域混合：平滑地融合变形后的区域

## 详细实现

### 1. 全局变形

```python
def deform_frame(self, frame, pose: PoseData) -> np.ndarray:
    """基于姿态数据变形图像"""
    height, width = frame.shape[:2]
    center = np.float32((width/2, height/2))
    
    # 1. 计算全局变换参数
    scale = self._calculate_scale(pose)      # 计算整体缩放
    rotation = self._calculate_rotation(pose) # 计算整体旋转
    translation = self._calculate_translation(pose, center) # 计算平移
    
    # 2. 构建变换矩阵
    M = cv2.getRotationMatrix2D(center, rotation, scale)
    M[:, 2] += translation
    
    # 3. 应用变换
    result = cv2.warpAffine(
        frame, M, (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
```

### 2. 区域变形

当提供了区域信息时，对每个区域单独进行变形：

```python
def _calculate_transform(self, region: DeformRegion, target_pose: PoseData):
    """计算区域变形矩阵"""
    # 1. 收集变形前后的对应点
    src_points = []  # 原始点
    dst_points = []  # 目标点
    
    for bp in region.binding_points:
        # 原始点 = 区域中心 + 局部坐标
        src_point = region.center + bp.local_coords
        src_points.append(src_point)
        
        # 目标点 = 目标姿态中对应关键点的位置
        landmark = target_pose.landmarks[bp.landmark_index]
        dst_point = np.array([landmark.x, landmark.y])
        dst_points.append(dst_point)
    
    # 2. 计算仿射变换矩阵
    transform = cv2.getAffineTransform(
        np.float32(src_points[:3]),
        np.float32(dst_points[:3])
    )
    return transform
```

### 3. 区域混合

使用权重混合不同区域的变形结果：

```python
def _blend_regions(self, frame, transformed_regions):
    """混合多个变形区域"""
    result = np.zeros_like(frame, dtype=float)
    weight_sum = np.zeros(frame.shape[:2], dtype=float)
    
    # 1. 累积每个区域的贡献
    for region in transformed_regions.values():
        # 计算区域权重
        weight = np.any(region > 0, axis=2).astype(float)
        weight = cv2.GaussianBlur(weight, (0,0), self._blend_radius)
        weight = np.expand_dims(weight, axis=2)
        
        # 累积变形结果和权重
        result += region * weight
        weight_sum += weight[..., 0]
    
    # 2. 归一化结果
    weight_sum = np.maximum(weight_sum, 1e-6)
    weight_sum = np.expand_dims(weight_sum, axis=2)
    result = result / weight_sum
    
    # 3. 处理未覆盖区域
    uncovered = weight_sum[..., 0] < 1e-6
    result[uncovered] = frame[uncovered]
    
    return result.astype(frame.dtype)
```

### 4. 时间平滑

为了避免抖动，对相邻帧进行平滑处理：

```python
# 在 deform_frame 中
if self._last_deformed is not None:
    result = cv2.addWeighted(
        self._last_deformed.astype(np.float32),
        self._smoothing_factor,
        result.astype(np.float32),
        1 - self._smoothing_factor,
        0
    ).astype(frame.dtype)

self._last_deformed = result.copy()
```

## 关键优化

1. 性能优化
- 使用 cv2.warpAffine 进行快速变形
- 并行处理多个区域的变形
- 使用 numpy 进行向量化计算

2. 质量优化
- 使用反射边界模式避免黑边
- 高斯模糊实现平滑过渡
- 时间平滑减少抖动

3. 内存优化
- 预分配结果数组
- 避免不必要的数组复制
- 及时释放临时变量

## 使用示例

```python
# 创建变形器
deformer = PoseDeformer(
    smoothing_window=5,
    blend_radius=20
)

# 应用变形
result = deformer.deform_frame(
    frame=input_frame,
    pose=target_pose
)
```

## 注意事项

1. 图像尺寸限制
- 最大支持 4096x4096 分辨率
- 最小需要 2x2 像素

2. 内存使用
- 峰值内存约为输入图像的 4-5 倍
- 建议预留 1GB 内存空间

3. 性能考虑
- 单帧处理时间应小于 33ms (30fps)
- 区域数量会影响性能 
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
# 变形平滑器 (DeformSmoother)

## 功能说明

DeformSmoother 是专门用于处理变形后图像的平滑器，继承自基础的 FrameSmoother。它主要解决以下问题：

1. 变形伪影处理
2. 边缘修复
3. 运动自适应平滑
4. 质量评估

## 技术原理

### 1. 变形区域检测

```python
def _detect_deform_regions(self, frame, original):
    """检测发生变形的区域"""
    # 1. 计算帧差
    diff = cv2.absdiff(frame, original)
    
    # 2. 阈值处理
    mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # 3. 形态学处理
    mask = cv2.dilate(mask, kernel)
```

### 2. 边缘修复

```python
def _repair_edges(self, frame, deform_mask):
    """修复变形边缘"""
    # 1. 边缘检测
    edges = cv2.Canny(frame, 50, 150)
    
    # 2. 获取需要修复的区域
    repair_mask = cv2.dilate(edges & deform_mask)
    
    # 3. 图像修复
    repaired = cv2.inpaint(frame, repair_mask)
```

### 3. 运动自适应

```python
def _adjust_smooth_params(self, frame, pose):
    """根据运动调整平滑参数"""
    # 1. 计算帧间运动
    frame_motion = mean_abs_diff(frame, prev_frame)
    
    # 2. 计算姿态运动
    pose_motion = norm(pose.velocity)
    
    # 3. 调整平滑参数
    smooth_weight = 0.8 - (frame_motion + pose_motion) * 0.5
```

## 使用方法

### 基本用法

```python
# 创建平滑器
smoother = DeformSmoother(
    model_path='models/NAFNet-GoPro-width32.pth',
    device='cuda'
)

# 处理单帧
smoothed = smoother.smooth_deformed_frame(
    frame=deformed_frame,
    original=original_frame,
    pose=pose_data
)
```

### 质量评估

```python
# 评估平滑质量
quality, details = smoother.assess_quality(smoothed_frame)

if quality < 0.7:
    logger.warning(f"平滑质量不佳: {quality:.2f}")
    logger.debug(f"详细评分: {details}")
```

## 参数配置

1. 变形检测
- deform_threshold: 变形检测阈值 (默认30)
- edge_width: 边缘修复宽度 (默认3)

2. 平滑控制
- temporal_weight: 时间平滑权重 (0.3-0.8)
- motion_scale: 运动影响系数 (默认0.5)

3. 质量评估
- quality_weights: 各项指标权重
  - temporal: 0.4
  - spatial: 0.3
  - edge: 0.3

## 性能说明

1. 计算开销
- 单帧处理时间: < 5ms (GPU)
- 内存占用: < 100MB

2. 质量指标
- 时间一致性: > 0.8
- 空间平滑度: > 0.7
- 边缘质量: > 0.9

## 注意事项

1. 使用建议
- 建议在变形后立即应用平滑
- 可以根据场景调整阈值
- 监控质量评分及时调整

2. 限制条件
- 输入图像尺寸需一致
- 需要连续帧才能时间平滑
- GPU版本需要CUDA支持 
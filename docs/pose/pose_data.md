# 姿态数据结构

## PoseData
姿态数据的主要容器类。

```python
@dataclass
class PoseData:
    landmarks: List[Landmark]    # 关键点列表
    timestamp: float            # 时间戳
    confidence: float          # 置信度
```

## Landmark
单个关键点的数据结构。

```python
@dataclass
class Landmark:
    x: float           # x坐标(归一化到0-1)
    y: float           # y坐标(归一化到0-1)
    z: float           # z坐标(相对深度)
    visibility: float  # 可见性(0-1)
```

## DeformRegion
变形区域的定义。

```python
@dataclass
class DeformRegion:
    center: np.ndarray          # 区域中心坐标
    binding_points: List[BindingPoint]  # 绑定点列表
    mask: Optional[np.ndarray]  # 区域蒙版
```

## BindingPoint
区域绑定点的定义。

```python
@dataclass
class BindingPoint:
    landmark_index: int        # 对应的关键点索引
    local_coords: np.ndarray   # 相对于中心点的局部坐标
    weight: float             # 影响权重(0-1)
```

## 数据转换
1. MediaPipe格式转换
   ```python
   keypoints = PoseDetector.mediapipe_to_keypoints(landmarks)
   pose_data = PoseData(keypoints=keypoints, timestamp=time.time())
   ```

2. 序列化/反序列化
   ```python
   # 序列化
   data = pose_data.to_dict()
   json_str = json.dumps(data)
   
   # 反序列化
   data = json.loads(json_str)
   pose_data = PoseData.from_dict(data)
   ```

## 注意事项
1. 坐标系统
   - 图像坐标: (0,0)在左上角
   - 归一化坐标: 范围0-1
   - 深度值: 相对值，无单位

2. 数据验证
   - 检查关键点完整性
   - 验证坐标范围
   - 确保权重和为1.0 
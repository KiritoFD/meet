# 姿态关键点定义

## 数据结构
```python
@dataclass
class PoseKeypoint:
    """姿态关键点定义"""
    id: int           # 关键点ID
    name: str         # 关键点名称
    parent_id: int    # 父关键点ID，-1表示无父节点
```

## 身体关键点
```python
POSE_KEYPOINTS = {
    # 躯干
    'nose': {'id': 0, 'name': 'nose', 'parent_id': -1},
    'neck': {'id': 1, 'name': 'neck', 'parent_id': 0},
    'right_shoulder': {'id': 12, 'name': 'right_shoulder', 'parent_id': 1},
    'left_shoulder': {'id': 11, 'name': 'left_shoulder', 'parent_id': 1},
    'right_hip': {'id': 24, 'name': 'right_hip', 'parent_id': 1},
    'left_hip': {'id': 23, 'name': 'left_hip', 'parent_id': 1},
    
    # 手臂
    'right_elbow': {'id': 14, 'name': 'right_elbow', 'parent_id': 12},
    'left_elbow': {'id': 13, 'name': 'left_elbow', 'parent_id': 11},
    'right_wrist': {'id': 16, 'name': 'right_wrist', 'parent_id': 14},
    'left_wrist': {'id': 15, 'name': 'left_wrist', 'parent_id': 13},
    
    # 腿部
    'right_knee': {'id': 26, 'name': 'right_knee', 'parent_id': 24},
    'left_knee': {'id': 25, 'name': 'left_knee', 'parent_id': 23},
    'right_ankle': {'id': 28, 'name': 'right_ankle', 'parent_id': 26},
    'left_ankle': {'id': 27, 'name': 'left_ankle', 'parent_id': 25}
}
```

## 连接定义
```python
POSE_CONNECTIONS = {
    'torso': ['left_shoulder', 'right_shoulder', 'right_hip', 'left_hip'],
    'right_arm': ['right_shoulder', 'right_elbow', 'right_wrist'],
    'left_arm': ['left_shoulder', 'left_elbow', 'left_wrist'],
    'right_leg': ['right_hip', 'right_knee', 'right_ankle'],
    'left_leg': ['left_hip', 'left_knee', 'left_ankle']
}
```

## 面部关键点
```python
FACE_CONNECTIONS = [
    # 眉毛
    ([70, 63, 105, 66, 107, 55, 65], 'eyebrow'),          # 左眉
    ([336, 296, 334, 293, 300, 285, 295], 'eyebrow'),     # 右眉
    
    # 眼睛
    ([33, 246, 161, 160, 159, 158, 157, 173, 133], 'eye'),  # 左眼
    ([362, 398, 384, 385, 386, 387, 388, 466, 263], 'eye'), # 右眼
    
    # 鼻子和嘴唇
    ([168, 6, 197, 195, 5], 'nose'),
    ([61, 185, 40, 39, 37, 0, 267, 269, 270, 409], 'mouth')
]
```

## 手部关键点
```python
HAND_CONNECTIONS = [
    # 拇指
    [0, 1, 2, 3, 4],
    # 食指
    [0, 5, 6, 7, 8],
    # 中指
    [0, 9, 10, 11, 12],
    # 无名指
    [0, 13, 14, 15, 16],
    # 小指
    [0, 17, 18, 19, 20]
]
```

## 关键点属性
1. 可见性(visibility)
   - 0.0: 不可见
   - 1.0: 完全可见
   - 中间值: 部分可见

2. 置信度(confidence)
   - 0.0: 完全不确定
   - 1.0: 完全确定
   - 阈值: 0.5

3. 坐标系统
   - x: 归一化横坐标 [0,1]
   - y: 归一化纵坐标 [0,1]
   - z: 相对深度值 [-1,1] 
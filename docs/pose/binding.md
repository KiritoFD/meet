# 姿态绑定 (PoseBinder)

## 技术原理

### 1. 关键点映射
```python
class PoseBinder:
    def __init__(self):
        # MediaPipe姿态关键点索引
        self.POSE_LANDMARKS = {
            'nose': 0,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28
        }
```

### 2. 区域划分算法
1. 基于骨骼的区域划分
```python
def segment_regions(self, landmarks):
    """基于骨骼结构划分区域
    
    - 躯干区域: 肩部和臀部关键点构成的四边形
    - 手臂区域: 肩部-手肘-手腕构成的三角形
    - 腿部区域: 臀部-膝盖-脚踝构成的三角形
    """
    regions = {
        'torso': self._create_torso_region(landmarks),
        'left_arm': self._create_arm_region(landmarks, 'left'),
        'right_arm': self._create_arm_region(landmarks, 'right'),
        'left_leg': self._create_leg_region(landmarks, 'left'),
        'right_leg': self._create_leg_region(landmarks, 'right')
    }
    return regions
```

2. 区域权重计算
```python
def calculate_weights(self, point, regions):
    """计算点到各区域的权重
    
    使用高斯权重函数:
    w = exp(-d^2 / (2σ^2))
    其中d为点到区域中心的距离
    """
    weights = {}
    for name, region in regions.items():
        d = np.linalg.norm(point - region.center)
        weights[name] = np.exp(-d**2 / (2 * self.sigma**2))
    return weights
```

### 3. 绑定点生成
```python
def generate_binding_points(self, frame_shape, regions):
    """生成区域绑定点
    
    1. 在每个区域内均匀采样点
    2. 计算每个点的局部坐标
    3. 关联最近的姿态关键点
    """
    binding_points = []
    for region in regions.values():
        # 网格采样
        points = self._grid_sample(region, spacing=20)
        
        # 计算局部坐标
        for point in points:
            local_coords = point - region.center
            nearest_landmark = self._find_nearest_landmark(point)
            binding_points.append(BindingPoint(
                landmark_index=nearest_landmark,
                local_coords=local_coords
            ))
    return binding_points
```

### 4. 变形约束
1. 骨骼长度保持
```python
def enforce_bone_length(self, landmarks):
    """保持骨骼长度不变"""
    bones = [
        ('left_shoulder', 'left_elbow'),
        ('left_elbow', 'left_wrist'),
        # ...其他骨骼
    ]
    for start, end in bones:
        self._adjust_bone_length(
            landmarks[self.POSE_LANDMARKS[start]],
            landmarks[self.POSE_LANDMARKS[end]]
        )
```

2. 关节角度限制
```python
def limit_joint_angles(self, landmarks):
    """限制关节活动范围"""
    joints = {
        'elbow': (-160, 0),  # 肘关节角度范围
        'knee': (-160, 0),   # 膝关节角度范围
        # ...其他关节
    }
    for joint, (min_angle, max_angle) in joints.items():
        self._clamp_joint_angle(landmarks, joint, min_angle, max_angle)
```

## 性能优化

### 1. 空间划分
使用四叉树加速最近点查找:
```python
class QuadTree:
    def __init__(self, bounds, max_points=4):
        self.bounds = bounds
        self.max_points = max_points
        self.points = []
        self.children = []
        
    def insert(self, point):
        """插入点"""
        if len(self.points) < self.max_points:
            self.points.append(point)
        else:
            if not self.children:
                self._split()
            self._insert_to_children(point)
            
    def find_nearest(self, query_point):
        """查找最近点"""
        if not self.children:
            return self._find_nearest_in_points(query_point)
        return min(
            (child.find_nearest(query_point) 
             for child in self.children),
            key=lambda p: np.linalg.norm(p - query_point)
        )
```

### 2. 并行处理
使用多线程处理区域变形:
```python
def deform_regions(self, frame, regions):
    """并行处理区域变形"""
    with ThreadPoolExecutor() as executor:
        futures = []
        for region in regions.values():
            future = executor.submit(
                self._deform_region, 
                frame, 
                region
            )
            futures.append(future)
        
        results = [f.result() for f in futures]
    return self._blend_results(results)
```

## 调试功能

### 1. 可视化工具
```python
def visualize_binding(self, frame, regions, binding_points):
    """可视化绑定结果"""
    vis = frame.copy()
    
    # 绘制区域
    for region in regions.values():
        cv2.polylines(vis, [region.contour], True, (0,255,0), 2)
        
    # 绘制绑定点
    for point in binding_points:
        cv2.circle(vis, tuple(point.coords), 3, (0,0,255), -1)
        
    # 绘制骨骼连接
    for start, end in self.bones:
        pt1 = tuple(map(int, landmarks[start]))
        pt2 = tuple(map(int, landmarks[end]))
        cv2.line(vis, pt1, pt2, (255,0,0), 2)
        
    return vis
```

### 2. 性能分析
```python
def profile_binding(self, frame, landmarks):
    """分析绑定性能"""
    import cProfile
    import pstats
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # 执行绑定
    self.bind(frame, landmarks)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats()
``` 
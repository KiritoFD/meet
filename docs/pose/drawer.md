# 姿态绘制器(PoseDrawer)

## 功能说明
负责将检测到的姿态关键点可视化绘制到图像上。

## 绘制配置
```python
DRAWER_CONFIG = {
    'colors': {
        'pose': (0, 255, 0),     # 姿态骨架颜色
        'face': (255, 0, 0),     # 面部特征颜色
        'hands': (0, 0, 255)     # 手部关键点颜色
    },
    'line_thickness': 2,         # 线条粗细
    'point_radius': 2,           # 关键点半径
    'text_scale': 0.5           # 文本大小
}
```

## API说明

### draw_frame()
```python
def draw_frame(self, frame: np.ndarray, results: Dict) -> np.ndarray:
    """绘制姿态检测结果
    
    Args:
        frame: 输入图像帧
        results: 检测结果，包含pose、face_mesh和hands
        
    Returns:
        np.ndarray: 绘制后的图像
    """
```

### draw_pose()
```python
def draw_pose(self, frame: np.ndarray, pose_results) -> None:
    """绘制身体姿态
    
    Args:
        frame: 输入图像帧
        pose_results: MediaPipe姿态检测结果
    """
```

### draw_face()
```python
def draw_face(self, frame: np.ndarray, face_results) -> None:
    """绘制面部特征
    
    Args:
        frame: 输入图像帧
        face_results: MediaPipe面部检测结果
    """
```

## 性能说明
- 绘制延迟 < 5ms
- 支持实时渲染(60fps)
- 内存占用稳定 
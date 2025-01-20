# 姿态检测配置

## MediaPipe配置
```python
MEDIAPIPE_CONFIG = {
    'pose': {
        'static_image_mode': False,
        'model_complexity': 1,
        'enable_segmentation': False,
        'smooth_landmarks': True,
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5
    },
    'hands': {
        'static_image_mode': False,
        'max_num_hands': 2,
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5
    },
    'face_mesh': {
        'static_image_mode': False,
        'max_num_faces': 1,
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5
    }
}
```

## 姿态处理配置
```python
POSE_CONFIG = {
    'detector': {
        'keypoints': {...},  # 关键点定义
        'connections': {...}  # 连接关系
    },
    'deformer': {
        'smoothing_window': 5,
        'smoothing_factor': 0.3,
        'blend_radius': 20,
        'min_scale': 0.5,
        'max_scale': 2.0
    },
    'drawer': {
        'colors': {...},      # 颜色方案
        'face_colors': {...}  # 面部颜色
    }
}
```

## 配置优化建议
1. 检测置信度
   - 提高置信度可增加稳定性
   - 降低置信度可提高检出率

2. 模型复杂度
   - 复杂度1: 平衡速度和准确性
   - 复杂度2: 更高准确性，更慢

3. 平滑设置
   - 增大平滑窗口可减少抖动
   - 减小平滑因子可提高响应速度 
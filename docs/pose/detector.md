# 姿态检测器(PoseDetector)

## 功能说明
使用 MediaPipe 进行人体姿态、手部和面部关键点的检测。

## 检测配置
```python
DETECTOR_CONFIG = {
    'pose': {
        'static_image_mode': False,
        'model_complexity': 1,
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

## API说明

### detect()
```python
def detect(self, frame: np.ndarray) -> Optional[Dict]:
    """检测单帧中的姿态
    
    Args:
        frame: 输入图像帧
        
    Returns:
        Dict: 包含pose、hands、face_mesh三种检测结果
        None: 如果检测失败
    """
    if frame is None:
        return None
        
    try:
        # 转换颜色空间
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 进行检测
        pose_results = self.pose.process(frame_rgb)
        hands_results = self.hands.process(frame_rgb)
        face_results = self.face_mesh.process(frame_rgb)
        
        return {
            'pose': pose_results,
            'hands': hands_results,
            'face_mesh': face_results
        }
        
    except Exception as e:
        logger.error(f"姿态检测失败: {str(e)}")
        return None
```

### mediapipe_to_keypoints()
```python
@classmethod
def mediapipe_to_keypoints(cls, landmarks) -> List[tuple]:
    """将MediaPipe姿态关键点转换为标准格式
    
    Args:
        landmarks: MediaPipe检测结果
        
    Returns:
        List[tuple]: [(id, x, y, z, visibility), ...]
    """
    keypoints = []
    for name, keypoint in cls.KEYPOINTS.items():
        lm = landmarks.landmark[keypoint.id]
        keypoints.append((
            keypoint.id,
            float(lm.x),
            float(lm.y),
            float(lm.z),
            float(lm.visibility)
        ))
    return keypoints
```

### release()
```python
def release(self):
    """释放检测器资源"""
    self.pose.close()
    self.hands.close()
    self.face_mesh.close()
```

## 性能说明
- 检测延迟 < 33ms
- 支持实时处理(30fps)
- GPU加速支持 
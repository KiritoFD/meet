
def detect(self, frame: np.ndarray) -> Optional[PoseData]:
    """检测单帧图像中的姿态"""
    if frame is None:
        return None
        
    try:
        # 转换为RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe处理
        results = self.pose.process(rgb_frame)
        if not results.pose_landmarks:
            return None
            
        # 提取关键点
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append({
                'x': float(lm.x),
                'y': float(lm.y),
                'z': float(lm.z),
                'visibility': float(lm.visibility)
            })
            
        # 创建姿态数据
        return PoseData(
            landmarks=landmarks,
            timestamp=time.time(),
            confidence=self._calculate_confidence(results)
        )
        
    except Exception as e:
        logger.error(f"姿态检测失败: {str(e)}")
        return None

# ...rest of the code...

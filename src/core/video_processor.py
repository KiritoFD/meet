import cv2
import mediapipe as mp
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class CaptureManager:
    def __init__(self):
        # 初始化摄像头等
        pass

    def start_capture(self):
        # 启动摄像头
        pass

    def release_capture(self):
        # 释放摄像头
        pass

class VideoProcessor:
    def __init__(self, capture_manager):
        self.capture_manager = capture_manager
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def process_frame(self, frame):
        # 使用 MediaPipe 处理帧
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)

        keypoints = []
        if results.pose_landmarks:
            selected_keypoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 22]
            for i in selected_keypoints:
                landmark = results.pose_landmarks.landmark[i]
                keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])

        return frame, keypoints
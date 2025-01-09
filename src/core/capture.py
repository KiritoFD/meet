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

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def process_frame_with_mediapipe(frame):
    # 将 BGR 图像转换为 RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 处理图像并获取结果
    results = pose.process(image)

    # 提取关键点
    keypoints = []
    if results.pose_landmarks:
        # 选择上半身关键点，特别是面部和手部
        selected_keypoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 22]
        for i in selected_keypoints:
            landmark = results.pose_landmarks.landmark[i]
            keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])

    # 将关键点信息添加到帧中（可选）
    # mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    return frame, keypoints
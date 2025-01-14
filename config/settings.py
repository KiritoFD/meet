import cv2

# 摄像头配置
CAMERA_CONFIG = {
    'api_preference': cv2.CAP_DSHOW,  # Windows上使用DirectShow
    'buffersize': 1,  # 减少缓冲
    'params': {
        cv2.CAP_PROP_FRAME_WIDTH: 640,
        cv2.CAP_PROP_FRAME_HEIGHT: 480,
        cv2.CAP_PROP_FPS: 30,
        cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*'MJPG'),
        cv2.CAP_PROP_BUFFERSIZE: 1
    }
}

# Flask 配置
FLASK_CONFIG = {
    'SECRET_KEY': 'your_secret_key_here',
    'DEBUG': False,
    'HOST': '0.0.0.0',
    'PORT': 5000
}

# MediaPipe 配置
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
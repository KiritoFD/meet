# 摄像头配置
CAMERA_CONFIG = {
    'index': 0,
    'width': 640,
    'height': 480,
    'fps': 24,
    'buffer_size': 1,
    'retry_interval': 0.05,
    'max_retries': 2
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
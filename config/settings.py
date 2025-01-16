import cv2

# 摄像头配置
CAMERA_CONFIG = {
    'api_preference': cv2.CAP_DSHOW,
    'device_id': 0,
    'params': {
        cv2.CAP_PROP_SETTINGS: 0,     # 禁用设置弹窗
        cv2.CAP_PROP_EXPOSURE: 15,     # 增加曝光值为3
        cv2.CAP_PROP_AUTOFOCUS: 0,    # 禁用自动对焦
        cv2.CAP_PROP_BUFFERSIZE: 1,   # 最小缓冲
        cv2.CAP_PROP_FPS: 30,         # 帧率
        cv2.CAP_PROP_FRAME_WIDTH: 640,  # 分辨率
        cv2.CAP_PROP_FRAME_HEIGHT: 480,
        cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*'MJPG')
    },
    'retry': {
        'max_attempts': 3,    # 最大重试次数
        'timeout': 3,         # 超时时间(秒)
        'interval': 0.1       # 重试间隔(秒)
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
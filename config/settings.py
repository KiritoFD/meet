import cv2

# 摄像头配置
CAMERA_CONFIG = {
    'api_preference': cv2.CAP_DSHOW,
    'device_id': 0,
    'params': {
        cv2.CAP_PROP_SETTINGS: 0,     # 禁用设置弹窗
        cv2.CAP_PROP_EXPOSURE: -3,    # 曝光值(尝试负值)
        cv2.CAP_PROP_BRIGHTNESS: 100,    # 增加亮度
        cv2.CAP_PROP_CONTRAST: 50,       # 对比度
        cv2.CAP_PROP_GAIN: 100,          # 增加增益
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
        'static_mode': False,
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

# 添加姿态检测和变形相关配置
POSE_CONFIG = {
    'detector': {
        'static_mode': False,
        'model_complexity': 1,
        'enable_segmentation': False,
        'smooth_landmarks': True,
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5,
        'min_confidence': 0.5,
        'smooth_factor': 0.5,
        
        'keypoints': {
            # 躯干
            'nose': {'id': 0, 'name': 'nose', 'parent_id': -1},
            'neck': {'id': 1, 'name': 'neck', 'parent_id': 0},
            'right_shoulder': {'id': 12, 'name': 'right_shoulder', 'parent_id': 1},
            'left_shoulder': {'id': 11, 'name': 'left_shoulder', 'parent_id': 1},
            'right_hip': {'id': 24, 'name': 'right_hip', 'parent_id': 1},
            'left_hip': {'id': 23, 'name': 'left_hip', 'parent_id': 1},
            
            # 手臂
            'right_elbow': {'id': 14, 'name': 'right_elbow', 'parent_id': 12},
            'left_elbow': {'id': 13, 'name': 'left_elbow', 'parent_id': 11},
            'right_wrist': {'id': 16, 'name': 'right_wrist', 'parent_id': 14},
            'left_wrist': {'id': 15, 'name': 'left_wrist', 'parent_id': 13},
            
            # 腿部
            'right_knee': {'id': 26, 'name': 'right_knee', 'parent_id': 24},
            'left_knee': {'id': 25, 'name': 'left_knee', 'parent_id': 23},
            'right_ankle': {'id': 28, 'name': 'right_ankle', 'parent_id': 26},
            'left_ankle': {'id': 27, 'name': 'left_ankle', 'parent_id': 25}
        },
        'connections': {
            'torso': ['left_shoulder', 'right_shoulder', 'right_hip', 'left_hip'],
            'left_upper_arm': ['left_shoulder', 'left_elbow'],
            'left_lower_arm': ['left_elbow', 'left_wrist'],
            'right_upper_arm': ['right_shoulder', 'right_elbow'],
            'right_lower_arm': ['right_elbow', 'right_wrist'],
            'left_upper_leg': ['left_hip', 'left_knee'],
            'left_lower_leg': ['left_knee', 'left_ankle'],
            'right_upper_leg': ['right_hip', 'right_knee'],
            'right_lower_leg': ['right_knee', 'right_ankle']
        }
    },
    'deformer': {
        'smoothing_window': 5,
        'smoothing_factor': 0.3,
        'blend_radius': 20,
        'min_scale': 0.5,
        'max_scale': 2.0,
        'control_point_radius': 50
    },
    'drawer': {
        'colors': {
            'face': (255, 0, 0),      # 蓝色
            'body': (0, 255, 0),      # 绿色
            'hands': (0, 255, 255),   # 黄色
            'joints': (0, 0, 255)     # 红色
        },
        'face_colors': {
            'contour': (200, 180, 130),    # 淡金色
            'eyebrow': (180, 120, 90),     # 深棕色
            'eye': (120, 150, 230),        # 淡蓝色
            'nose': (150, 200, 180),       # 青绿色
            'mouth': (140, 160, 210)       # 淡紫色
        }
    },
    'smoother': {
        # 基础平滑参数
        'temporal_weight': 0.8,
        'spatial_weight': 0.5,
        
        # 变形平滑参数
        'deform_threshold': 30,
        'edge_width': 3,
        'motion_scale': 0.5,
        
        # 质量评估参数
        'quality_weights': {
            'temporal': 0.4,
            'spatial': 0.3,
            'edge': 0.3
        }
    }
} 
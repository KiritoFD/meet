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
        'static_image_mode': False,  # 修改 'static_mode' 为 'static_image_mode'
        'model_complexity': 2,
        'enable_segmentation': True,
        'smooth_landmarks': True,
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5
    },
    'hands': {
        'static_image_mode': False,
        'max_num_hands': 2,
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5,
        'model_complexity': 1
    },
    'face_mesh': {
        'static_image_mode': False,
        'max_num_faces': 1,
        'refine_landmarks': True,  # 启用细节关键点
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5
    }
}

# 添加姿态检测和变形相关配置
POSE_CONFIG = {
    'detector': {
        # 基本参数
        'static_mode': False,
        'model_complexity': 2,
        'smooth_landmarks': True,
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5,
        'min_confidence': 0.3,  # 降低置信度阈值以提高检测率
        'smooth_factor': 0.5,
        
        # 身体关键点定义 - 移到前面来
        'body_landmarks': {
            'nose': 0,
            'left_eye_inner': 1,
            'left_eye': 2,
            'left_eye_outer': 3,
            'right_eye_inner': 4,
            'right_eye': 5,
            'right_eye_outer': 6,
            'left_ear': 7,
            'right_ear': 8,
            'mouth_left': 9,
            'mouth_right': 10,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_pinky': 17,
            'right_pinky': 18,
            'left_index': 19,
            'right_index': 20,
            'left_thumb': 21,
            'right_thumb': 22,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
            'left_heel': 29,
            'right_heel': 30,
            'left_foot_index': 31,
            'right_foot_index': 32
        },

        # 面部关键点 - 移到前面来
        'face_landmarks': {
            'contour': list(range(0, 17)) + list(range(297, 318)),  # 脸部轮廓
            'left_eye': list(range(362, 374)),      # 左眼
            'right_eye': list(range(133, 145)),     # 右眼
            'left_eyebrow': list(range(276, 283)),  # 左眉毛
            'right_eyebrow': list(range(46, 53)),   # 右眉毛
            'nose': list(range(168, 175)),          # 鼻子
            'mouth_outer': list(range(0, 17)),      # 外唇
            'mouth_inner': list(range(78, 87))      # 内唇
        },

        # 手部关键点 - 移到前面来
        'hand_landmarks': {
            'wrist': 0,
            'thumb': list(range(1, 5)),
            'index_finger': list(range(5, 9)),
            'middle_finger': list(range(9, 13)),
            'ring_finger': list(range(13, 17)),
            'pinky': list(range(17, 21))
        },

        # 关键点定义 - 现在可以安全引用上面定义的部分
        'keypoints': {
            'body': 'body_landmarks',  # 使用字符串引用
            'face': 'face_landmarks',
            'hands': 'hand_landmarks'
        },

        # 关键点连接定义
        'connections': {
            # 面部连接
            'face': {
                'contour': [(i, i+1) for i in range(0, 16)] + [(i, i+1) for i in range(297, 317)],
                'left_eye': [(i, i+1) for i in range(362, 373)] + [(373, 362)],
                'right_eye': [(i, i+1) for i in range(133, 144)] + [(144, 133)],
                'left_eyebrow': [(i, i+1) for i in range(276, 282)],
                'right_eyebrow': [(i, i+1) for i in range(46, 52)],
                'nose': [(i, i+1) for i in range(168, 174)],
                'mouth': [(i, i+1) for i in range(0, 16)] + [(0, 16)]
            },
            
            # 手部连接
            'hands': {
                'thumb': [(0,1), (1,2), (2,3), (3,4)],
                'index': [(0,5), (5,6), (6,7), (7,8)],
                'middle': [(0,9), (9,10), (10,11), (11,12)],
                'ring': [(0,13), (13,14), (14,15), (15,16)],
                'pinky': [(0,17), (17,18), (18,19), (19,20)]
            },
            
            # 身体连接
            'body': {
                'torso': [
                    (11,12), (11,23), (12,24), (23,24),  # 躯干
                    (11,13), (13,15), (12,14), (14,16),  # 手臂
                    (23,25), (25,27), (24,26), (26,28)   # 腿部
                ],
                'face': [
                    (0,1), (1,2), (2,3), (3,7),          # 左侧脸
                    (0,4), (4,5), (5,6), (6,8),          # 右侧脸
                    (9,10)                                # 嘴
                ],
                'feet': [
                    (27,29), (29,31), (28,30), (30,32)   # 脚部
                ]
            }
        }
    },
    
    # 变形器配置
    'deformer': {
        'smoothing_window': 5,
        'smoothing_factor': 0.3,
        'blend_radius': 20,
        'min_scale': 0.5,
        'max_scale': 2.0,
        'control_point_radius': 50,
        'interpolation_method': 'linear',
        'edge_preservation': True,
        'motion_threshold': 0.1
    },
    
    # 绘制器配置
    'drawer': {
        'colors': {
            'face': (255, 200, 0),    # 淡蓝色
            'body': (0, 255, 0),      # 绿色
            'hands': (0, 255, 255),   # 黄色
            'joints': (255, 0, 0)     # 红色
        },
        'line_thickness': {
            'face': 1,
            'body': 2,
            'hands': 1
        },
        'point_size': {
            'face': 1,
            'body': 3,
            'hands': 2
        }
    }
}
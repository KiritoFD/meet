import cv2
import numpy as np
import mediapipe as mp
import time

class FacialExpressionMapper:
    def __init__(self):
        # 初始化 MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 定义关键面部特征点
        self.FACIAL_FEATURES = {
            'left_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],  # 左眼完整轮廓
            'right_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],   # 右眼完整轮廓
            'mouth': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146], # 嘴巴完整轮廓
            'left_eyebrow': [276, 283, 282, 295, 285, 300, 293, 334, 296],  # 左眉毛完整轮廓
            'right_eyebrow': [46, 53, 52, 65, 55, 70, 63, 105, 66],         # 右眉毛完整轮廓
            'nose': [168, 197, 5, 4, 1, 19, 94, 2, 164, 165, 167]  # 更新的鼻子轮廓点
        }

    def draw_facial_features(self, frame, face_landmarks):
        """绘制面部特征点和轮廓"""
        frame_height, frame_width = frame.shape[:2]
        
        # 为每个特征定义不同的颜色和连接方式
        for feature, points in self.FACIAL_FEATURES.items():
            # 获取特征点坐标
            coords = []
            for point in points:
                landmark = face_landmarks.landmark[point]
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                coords.append((x, y))
            
            # 根据不同特征使用不同的颜色和绘制方式
            if 'eye' in feature:
                color = (255, 0, 0)  # 蓝色
                cv2.polylines(frame, [np.array(coords)], True, color, 1)
            elif 'mouth' in feature:
                color = (0, 0, 255)  # 红色
                cv2.polylines(frame, [np.array(coords)], True, color, 1)
            elif 'eyebrow' in feature:
                color = (0, 255, 255)  # 黄色
                cv2.polylines(frame, [np.array(coords)], False, color, 1)
            elif 'nose' in feature:
                color = (0, 255, 0)  # 绿色
                # 鼻子特征采用点的方式显示，不连线
                for x, y in coords:
                    cv2.circle(frame, (x, y), 2, color, -1)
            
            # 绘制关键点
            for x, y in coords:
                cv2.circle(frame, (x, y), 1, color, -1)
        
        return frame

    def get_feature_states(self, landmarks):
        """获取面部特征状态"""
        points = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks.landmark])
        
        # 计算各个特征的状态
        states = {
            'left_eye_openness': self.calculate_eye_openness([points[i] for i in self.FACIAL_FEATURES['left_eye']]),
            'right_eye_openness': self.calculate_eye_openness([points[i] for i in self.FACIAL_FEATURES['right_eye']]),
            'mouth_openness': self.calculate_mouth_openness([points[i] for i in self.FACIAL_FEATURES['mouth']]),
            'mouth_width': self.calculate_mouth_width([points[i] for i in self.FACIAL_FEATURES['mouth']]),
            'eyebrow_height': {
                'left': np.mean([points[i][1] for i in self.FACIAL_FEATURES['left_eyebrow']]),
                'right': np.mean([points[i][1] for i in self.FACIAL_FEATURES['right_eyebrow']])
            }
        }
        return states

    def calculate_eye_openness(self, eye_points):
        """计算眼睛开合度"""
        height = np.mean([eye_points[1][1], eye_points[2][1]]) - np.mean([eye_points[4][1], eye_points[5][1]])
        width = eye_points[3][0] - eye_points[0][0]
        return height / width if width != 0 else 0

    def calculate_mouth_openness(self, mouth_points):
        """计算嘴巴开合度"""
        height = np.mean([mouth_points[3][1], mouth_points[4][1]]) - np.mean([mouth_points[9][1], mouth_points[10][1]])
        width = mouth_points[6][0] - mouth_points[0][0]
        return height / width if width != 0 else 0

    def calculate_mouth_width(self, mouth_points):
        """计算嘴巴宽度"""
        return mouth_points[6][0] - mouth_points[0][0]

    def process_frame(self, frame):
        """处理单帧图像"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            # 获取面部特征状态
            states = self.get_feature_states(results.multi_face_landmarks[0])
            
            # 绘制面部特征
            frame = self.draw_facial_features(frame, results.multi_face_landmarks[0])
            
            # 显示特征状态数据
            y_pos = 30
            for feature, value in states.items():
                if isinstance(value, dict):
                    for side, v in value.items():
                        text = f"{feature}_{side}: {v:.2f}"
                        cv2.putText(frame, text, (10, y_pos),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                  (200, 200, 200), 1)
                        y_pos += 20
                else:
                    text = f"{feature}: {value:.2f}"
                    cv2.putText(frame, text, (10, y_pos),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                              (200, 200, 200), 1)
                    y_pos += 20
                
        return frame

def test_webcam():
    """使用摄像头测试"""
    mapper = FacialExpressionMapper()
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("正在启动摄像头测试...")
    print("按 'q' 退出")
    
    fps_time = time.time()
    fps_frames = 0
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame = mapper.process_frame(frame)
        
        fps_frames += 1
        if time.time() - fps_time > 1:
            fps = fps_frames
            fps_frames = 0
            fps_time = time.time()
        
        cv2.putText(
            processed_frame,
            f"FPS: {fps}",
            (540, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )
        
        cv2.imshow('Facial Feature Mapping', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_webcam() 
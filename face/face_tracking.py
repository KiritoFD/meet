import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
import json

@dataclass
class FacialFeatures:
    """面部特征数据结构"""
    landmarks: np.ndarray  # 特征点坐标
    timestamp: float       # 时间戳

class FaceTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # 用于存储初始帧数据
        self.base_features = None
        self.personal_model = None
    
    def extract_features(self, frame) -> FacialFeatures:
        """提取面部特征点"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        # 提取特征点坐标
        landmarks = np.array([[lm.x, lm.y, lm.z] 
                            for lm in results.multi_face_landmarks[0].landmark])
        
        return FacialFeatures(
            landmarks=landmarks,
            timestamp=cv2.getTickCount() / cv2.getTickFrequency()
        )
    
    def capture_base_features(self, frames_count=30):
        """捕获用户的基础特征（用于生成个人模型）"""
        print("正在捕获基础特征...")
        cap = cv2.VideoCapture(0)
        features_list = []
        
        while len(features_list) < frames_count:
            ret, frame = cap.read()
            if not ret:
                break
                
            features = self.extract_features(frame)
            if features is not None:
                features_list.append(features)
                print(f"已捕获 {len(features_list)}/{frames_count} 帧")
            
            # 显示预览
            cv2.imshow('Capturing Base Features', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if features_list:
            self.base_features = features_list
            print("基础特征捕获完成！")
            return True
        return False
    
    def serialize_features(self, features: FacialFeatures) -> bytes:
        """序列化特征数据（用于网络传输）"""
        data = {
            'landmarks': features.landmarks.tolist(),
            'timestamp': features.timestamp
        }
        return json.dumps(data).encode()
    
    def deserialize_features(self, data: bytes) -> FacialFeatures:
        """反序列化特征数据"""
        data_dict = json.loads(data.decode())
        return FacialFeatures(
            landmarks=np.array(data_dict['landmarks']),
            timestamp=data_dict['timestamp']
        )
    
    def visualize_features(self, frame, features: FacialFeatures):
        """可视化特征点（调试用）"""
        result = frame.copy()
        h, w = frame.shape[:2]
        
        # 绘制特征点
        for i, (x, y, z) in enumerate(features.landmarks):
            px = int(x * w)
            py = int(y * h)
            # 使用z值来决定颜色（深度信息）
            color = (int(255 * (1-z)), int(255 * z), 0)
            cv2.circle(result, (px, py), 1, color, -1)
            
        # 显示数据大小
        data_size = len(self.serialize_features(features))
        cv2.putText(result, f'Data size: {data_size} bytes', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return result

def main():
    tracker = FaceTracker()
    
    # 首先捕获基础特征（用于生成个人模型）
    if not tracker.capture_base_features():
        print("无法捕获基础特征")
        return
    
    # 开始实时追踪
    cap = cv2.VideoCapture(0)
    print("\n开始实时追踪")
    print("按 'q' 退出")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 提取特征
        features = tracker.extract_features(frame)
        if features is not None:
            # 序列化（模拟网络传输）
            data = tracker.serialize_features(features)
            
            # 反序列化（模拟接收端）
            received_features = tracker.deserialize_features(data)
            
            # 可视化
            result = tracker.visualize_features(frame, received_features)
            cv2.imshow('Face Tracking', result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
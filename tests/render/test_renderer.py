import cv2
import mediapipe as mp
import numpy as np

class SimpleAnimeRenderer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
    def process_frame(self, frame):
        # 转换颜色空间
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return frame
            
        # 获取图像尺寸
        height, width = frame.shape[:2]
        
        # 创建遮罩
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 绘制面部区域
        face_points = []
        for landmark in results.multi_face_landmarks[0].landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            face_points.append([x, y])
        
        face_points = np.array(face_points, dtype=np.int32)
        hull = cv2.convexHull(face_points)
        cv2.fillConvexPoly(mask, hull, 255)
        
        # 应用双边滤波进行磨皮
        filtered = cv2.bilateralFilter(frame, 9, 75, 75)
        
        # 应用CLAHE增强对比度
        lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 检测边缘
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # 合并结果
        result = frame.copy()
        result = cv2.bitwise_and(enhanced, enhanced, mask=mask)
        bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
        result = cv2.add(result, bg)
        
        # 添加边缘
        result = cv2.addWeighted(result, 0.8, edges, 0.2, 0)
        
        # 绘制特征点（调试用）
        for landmark in results.multi_face_landmarks[0].landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(result, (x, y), 1, (0, 255, 0), -1)
        
        return result

def main():
    # 初始化渲染器
    renderer = SimpleAnimeRenderer()
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    print("按 'q' 退出")
    print("按 's' 保存当前帧")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 处理帧
        result = renderer.process_frame(frame)
        
        # 显示原始图像和结果
        cv2.imshow('Original', frame)
        cv2.imshow('Anime Style', result)
        
        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('original.jpg', frame)
            cv2.imwrite('anime_style.jpg', result)
            print("图片已保存！")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
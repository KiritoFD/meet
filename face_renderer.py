import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass

@dataclass
class FaceFeatures:
    """面部特征数据类"""
    landmarks: np.ndarray  # 468个面部特征点
    contours: dict  # 面部轮廓（眼睛、嘴巴等）
    mesh: np.ndarray  # 面部网格
    texture_coords: np.ndarray  # 纹理坐标

class AnimeStyleRenderer:
    def __init__(self):
        # 初始化 MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,  # 实时模式
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # 定义关键面部区域的索引
        self.FACE_CONTOURS = {
            'face_oval': list(self.mp_face_mesh.FACEMESH_FACE_OVAL),
            'lips': list(self.mp_face_mesh.FACEMESH_LIPS),
            'left_eye': list(self.mp_face_mesh.FACEMESH_LEFT_EYE),
            'right_eye': list(self.mp_face_mesh.FACEMESH_RIGHT_EYE),
            'eyebrows': list(self.mp_face_mesh.FACEMESH_EYEBROWS)
        }
        
        # 加载基础动漫风格纹理
        self.base_texture = cv2.imread('assets/anime_base_texture.png')
        
    def extract_features(self, image) -> FaceFeatures:
        """提取面部特征"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return None
            
        landmarks = results.multi_face_landmarks[0]
        
        # 转换landmarks为numpy数组
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        
        # 提取面部轮廓
        contours = {}
        for name, indices in self.FACE_CONTOURS.items():
            contours[name] = points[indices]
            
        # 生成面部网格
        mesh = self._generate_mesh(points)
        
        # 计算纹理坐标
        texture_coords = self._calculate_texture_coords(points)
        
        return FaceFeatures(points, contours, mesh, texture_coords)
    
    def _generate_mesh(self, points):
        """生成面部三角网格"""
        # 使用Delaunay三角剖分
        hull = cv2.convexHull(points[:, :2].astype(np.float32))
        rect = cv2.boundingRect(hull)
        subdiv = cv2.Subdiv2D(rect)
        
        for point in points[:, :2]:
            subdiv.insert(tuple(map(float, point)))
            
        triangles = subdiv.getTriangleList()
        return triangles
    
    def _calculate_texture_coords(self, points):
        """计算纹理映射坐标"""
        # 将3D点映射到UV空间
        min_point = points.min(axis=0)
        max_point = points.max(axis=0)
        normalized = (points - min_point) / (max_point - min_point)
        return normalized[:, :2]  # 只需要x,y坐标
    
    def render(self, image, features: FaceFeatures):
        """渲染动漫风格"""
        if features is None:
            return image
            
        result = image.copy()
        
        # 1. 平滑肤色
        mask = self._create_face_mask(image.shape[:2], features.contours['face_oval'])
        skin = cv2.bilateralFilter(result, 9, 75, 75)
        result = cv2.bitwise_and(skin, skin, mask=mask)
        
        # 2. 增强边缘（卡通效果）
        edges = self._detect_edges(image, features)
        result = self._apply_edges(result, edges)
        
        # 3. 应用动漫风格色彩
        result = self._apply_anime_colors(result, features)
        
        return result
    
    def _create_face_mask(self, shape, contour):
        """创建面部遮罩"""
        mask = np.zeros(shape, dtype=np.uint8)
        contour = (contour[:, :2] * shape).astype(np.int32)
        cv2.fillPoly(mask, [contour], 255)
        return mask
    
    def _detect_edges(self, image, features):
        """检测面部边缘"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return edges
    
    def _apply_edges(self, image, edges):
        """应用边缘到图像"""
        return cv2.addWeighted(image, 0.8, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.2, 0)
    
    def _apply_anime_colors(self, image, features):
        """应用动漫风格的颜色"""
        # 使用LAB色彩空间进行颜色调整
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 增强对比度
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # 调整色相和饱和度
        merged = cv2.merge([l, a, b])
        result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        
        return result

def main():
    # 初始化渲染器
    renderer = AnimeStyleRenderer()
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 提取面部特征
        features = renderer.extract_features(frame)
        
        # 渲染动漫风格
        if features is not None:
            result = renderer.render(frame, features)
            cv2.imshow('Anime Style', result)
        else:
            cv2.imshow('Anime Style', frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
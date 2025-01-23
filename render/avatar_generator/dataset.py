import cv2
import numpy as np
from pathlib import Path
import dlib
from PIL import Image
import torch
from torchvision import transforms
from sklearn.cluster import KMeans
import mediapipe as mp

class ImageProcessor:
    def __init__(self):
        # 初始化各种检测器
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
        # dlib的特征点检测器
        self.face_detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            'models/shape_predictor_68_face_landmarks.dat'
        )
        
        # 图像增强和标准化
        self.transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def detect_face_quality(self, image):
        """评估面部图像质量"""
        quality_score = 0
        
        # 检查图像清晰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var > 100:  # 清晰度阈值
            quality_score += 1
        
        # 检查光照条件
        brightness = np.mean(gray)
        if 50 < brightness < 200:  # 合适的亮度范围
            quality_score += 1
        
        # 检查对比度
        contrast = np.std(gray)
        if contrast > 20:  # 对比度阈值
            quality_score += 1
        
        return quality_score >= 2  # 至少满足两个条件

    def align_face(self, image, landmarks):
        """根据面部特征点对齐面部"""
        # 计算眼睛中心点
        left_eye = np.mean(landmarks[36:42], axis=0)
        right_eye = np.mean(landmarks[42:48], axis=0)
        
        # 计算角度
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # 旋转图像
        center = (image.shape[1]//2, image.shape[0]//2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned_image = cv2.warpAffine(
            image, 
            rotation_matrix, 
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_CUBIC
        )
        
        return aligned_image

    def enhance_face(self, image):
        """增强面部图像质量"""
        # 自适应直方图均衡化
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 降噪
        enhanced = cv2.fastNlMeansDenoisingColored(
            enhanced,
            None,
            10,
            10,
            7,
            21
        )
        
        return enhanced

    def process_image(self, img_path, size=1024):
        """改进的图像处理流程"""
        # 读取图像
        image = cv2.imread(str(img_path))
        if image is None:
            return None
            
        # 1. 质量检查
        if not self.detect_face_quality(image):
            return None
        
        # 2. 面部检测和特征点提取
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)
        if len(faces) == 0:
            return None
        
        # 获取面部特征点
        shape = self.predictor(gray, faces[0])
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        
        # 3. 面部对齐
        aligned_image = self.align_face(image, landmarks)
        
        # 4. 裁剪面部区域
        face_rect = faces[0]
        center_x = (face_rect.left() + face_rect.right()) // 2
        center_y = (face_rect.top() + face_rect.bottom()) // 2
        
        # 扩大裁剪区域以包含更多上下文
        box_size = int(max(face_rect.width(), face_rect.height()) * 1.5)
        left = max(0, center_x - box_size//2)
        top = max(0, center_y - box_size//2)
        right = min(image.shape[1], center_x + box_size//2)
        bottom = min(image.shape[0], center_y + box_size//2)
        
        face = aligned_image[top:bottom, left:right]
        
        # 5. 图像增强
        enhanced_face = self.enhance_face(face)
        
        # 6. 调整大小
        resized_face = cv2.resize(enhanced_face, (size, size))
        
        # 7. 颜色校正
        # 使用K-means进行主色调分析
        pixels = resized_face.reshape(-1, 3)
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(pixels)
        
        # 获取主色调
        dominant_colors = kmeans.cluster_centers_
        
        # 根据主色调调整色彩平衡
        if np.mean(dominant_colors) < 100:  # 如果整体偏暗
            resized_face = cv2.convertScaleAbs(
                resized_face, 
                alpha=1.2, 
                beta=10
            )
        
        return resized_face

class DatasetManager:
    def __init__(self, base_path='datasets'):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.raw_path = self.base_path / 'raw'
        self.processed_path = self.base_path / 'processed'
        self.raw_path.mkdir(exist_ok=True)
        self.processed_path.mkdir(exist_ok=True)
        
        # 使用改进的图像处理器
        self.image_processor = ImageProcessor()
    
    def process_dataset(self, input_dir=None, size=1024):
        """使用改进的处理方法处理数据集"""
        if input_dir is None:
            input_dir = self.raw_path
        
        input_dir = Path(input_dir)
        output_dir = self.processed_path / input_dir.name
        output_dir.mkdir(exist_ok=True)
        
        for img_path in tqdm(list(input_dir.glob('*.jpg')), desc="处理图片"):
            processed = self.image_processor.process_image(img_path, size)
            if processed is not None:
                output_path = output_dir / img_path.name
                cv2.imwrite(str(output_path), processed)

    # ... 其他方法保持不变 ... 
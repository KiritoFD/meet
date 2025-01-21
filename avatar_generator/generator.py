import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import mediapipe as mp
import os
from .models.anime_gan import AnimeGANv2Generator

class SimpleAnimeGenerator(nn.Module):
    def __init__(self):
        super(SimpleAnimeGenerator, self).__init__()
        
        # 简化的生成器网络
        self.main = nn.Sequential(
            # 下采样
            nn.Conv2d(3, 64, 7, 1, 3),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            
            # 残差块
            *[ResBlock(256) for _ in range(6)],
            
            # 上采样
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            
            # 输出层
            nn.Conv2d(64, 3, 7, 1, 3),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.main(x)

class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class AvatarGenerator:
    def __init__(self, model_path='models/animeganv2.pth'):
        # 初始化 MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # 初始化模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AnimeGANv2Generator().to(self.device)
        
        if model_path and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                
                # 处理模型键名
                if 'generator' in state_dict:
                    state_dict = state_dict['generator']
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
                self.model.load_state_dict(state_dict, strict=False)
                print("模型加载成功！")
            except Exception as e:
                print(f"加载模型失败: {e}")
                print("使用未训练的模型...")
        else:
            print("未找到模型文件，使用未训练的模型...")
        
        self.model.eval()
        
        # 更新图像预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def capture_reference(self):
        """从摄像头捕捉参考照片"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
            return None
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        reference_image = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 显示实时预览
            preview_frame = frame.copy()
            
            # 检测人脸
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                # 获取人脸边界框
                h, w = frame.shape[:2]
                landmarks = results.multi_face_landmarks[0].landmark
                
                x_min = w
                y_min = h
                x_max = 0
                y_max = 0
                
                for landmark in landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                
                # 绘制人脸框
                cv2.rectangle(preview_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # 检查人脸位置
                face_width = x_max - x_min
                face_height = y_max - y_min
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                
                if (0.3 * w < face_width < 0.7 * w and
                    abs(center_x - w/2) < w * 0.1 and
                    abs(center_y - h/2) < h * 0.1):
                    cv2.putText(preview_frame, "Position OK - Press SPACE", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if key == ord(' '):  # 空格键拍照
                        reference_image = frame.copy()
                        break
                else:
                    cv2.putText(preview_frame, "Adjust Position", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Camera', preview_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # q键退出
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return reference_image

    def generate_avatar(self, image):
        """生成动漫风格头像"""
        if image is None:
            return None
            
        try:
            # 预处理
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
            
            # 生成
            with torch.no_grad():
                output_tensor = self.model(input_tensor)
            
            # 后处理
            output = output_tensor[0].cpu().numpy()
            output = (output * 0.5 + 0.5) * 255
            output = output.transpose(1, 2, 0)
            output = output.astype(np.uint8)
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            
            return output
        except Exception as e:
            print(f"生成失败: {e}")
            return None 
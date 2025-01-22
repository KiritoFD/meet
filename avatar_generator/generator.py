import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import mediapipe as mp
import os
from models.anime_gan import AnimeGANv2Generator
import time
from face.face_verification import FaceVerifier

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
        """从摄像头捕捉参考照片并进行人脸验证"""
        # 检查摄像头是否可用
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("错误：摄像头未打开，请先打开摄像头")
            return None
            
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 初始化人脸验证器
        face_verifier = FaceVerifier(similarity_threshold=0.6)
        reference_image = None
        verification_passed = False
        
        print("请面对摄像头保持自然表情...")
        print("按空格键拍照，按q退出")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("错误：无法读取摄像头画面")
                break
                
            # 显示实时预览
            preview_frame = frame.copy()
            
            # 检测人脸
            face_location = face_verifier.get_face_location(frame)
            
            if face_location:
                # 获取人脸边界框
                top, right, bottom, left = face_location
                
                # 检查人脸位置和大小是否合适
                face_width = right - left
                face_height = bottom - top
                center_x = (left + right) / 2
                center_y = (top + bottom) / 2
                
                if (0.3 * frame.shape[1] < face_width < 0.7 * frame.shape[1] and
                    abs(center_x - frame.shape[1]/2) < frame.shape[1] * 0.1 and
                    abs(center_y - frame.shape[0]/2) < frame.shape[0] * 0.1):
                    
                    # 绘制绿色边框表示位置合适
                    cv2.rectangle(preview_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(preview_frame, "Position OK - Press SPACE", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # 检查按键
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' '):  # 空格键拍照
                        # 设置为参考帧
                        if face_verifier.set_reference(frame):
                            reference_image = frame.copy()
                            print("参考帧已捕获，请稍等...")
                            
                            # 等待2秒进行验证
                            time.sleep(2)
                            
                            # 进行人脸验证
                            result = face_verifier.verify_face(frame)
                            if result.is_same_person:
                                print(f"验证通过！相似度: {result.confidence:.2f}")
                                verification_passed = True
                                break
                            else:
                                print(f"验证失败，请重试。相似度: {result.confidence:.2f}")
                                face_verifier.clear_reference()
                        else:
                            print("未能正确捕获人脸，请重试")
                else:
                    # 绘制红色边框表示需要调整位置
                    cv2.rectangle(preview_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(preview_frame, "Adjust Position", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(preview_frame, "No Face Detected", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Camera', preview_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):  # q键退出
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if verification_passed:
            return reference_image
        else:
            print("参考帧捕获失败或验证未通过")
            return None

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
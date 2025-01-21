import subprocess
import sys
import os

def setup_animegan():
    """设置 AnimeGAN 环境"""
    print("正在设置 AnimeGAN 环境...")
    
    # 检查是否已经克隆了仓库
    if not os.path.exists('animegan2-pytorch'):
        print("克隆 AnimeGAN 仓库...")
        try:
            subprocess.check_call(['git', 'clone', 'https://github.com/bryandlee/animegan2-pytorch.git'])
        except subprocess.CalledProcessError as e:
            print(f"克隆仓库失败: {e}")
            return False
    
    # 将仓库目录添加到 Python 路径
    repo_path = os.path.abspath('animegan2-pytorch')
    if repo_path not in sys.path:
        sys.path.append(repo_path)
    
    return True

def install_requirements():
    """安装必要的包"""
    print("正在安装必要的包...")
    requirements = [
        "torch",
        "torchvision",
        "opencv-python",
        "mediapipe",
        "Pillow",
    ]
    
    for package in requirements:
        print(f"安装 {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"安装 {package} 失败: {e}")
            return False
    print("所有依赖安装完成！")
    return True

if __name__ == "__main__":
    # 先安装依赖
    if not install_requirements():
        print("依赖安装失败，程序退出")
        sys.exit(1)
    
    # 设置 AnimeGAN 环境
    if not setup_animegan():
        print("AnimeGAN 环境设置失败，程序退出")
        sys.exit(1)
    
    # 现在可以安全地导入其他模块
    import cv2
    import torch
    from PIL import Image
    import numpy as np
    from torchvision.transforms.functional import to_tensor, to_pil_image
    from model import Generator  # 从克隆的仓库导入
    import mediapipe as mp

    def capture_photo():
        """从摄像头捕捉照片"""
        # 初始化人脸检测
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
            return None
        
        print("请面对摄像头，保持自然表情")
        print("按空格键拍照，按q键退出")
        
        captured_frame = None  # 初始化捕获的帧
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 显示实时预览
            preview = frame.copy()
            
            # 检测人脸
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            
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
                cv2.rectangle(preview, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # 检查人脸位置
                face_width = x_max - x_min
                face_height = y_max - y_min
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                
                if (0.3 * w < face_width < 0.7 * w and
                    abs(center_x - w/2) < w * 0.1 and
                    abs(center_y - h/2) < h * 0.1):
                    cv2.putText(preview, "Position OK - Press SPACE", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(preview, "Adjust Position", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Camera', preview)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # 空格键拍照
                if results.multi_face_landmarks:
                    captured_frame = frame.copy()
                    print("照片已捕捉！")
                    break
            elif key == ord('q'):  # q键退出
                print("已取消拍照")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return captured_frame

    def main():
        # 捕捉照片
        image = capture_photo()
        if image is None:
            print("未能捕捉到照片")
            return
            
        # 保存原始照片
        cv2.imwrite('original.jpg', image)
        
        # 转换为PIL图像
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        # 加载模型
        print("正在加载模型...")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = Generator()
        model = model.to(device)
        
        # 加载预训练权重
        pretrained_path = 'animegan2-pytorch/weights/face_paint_512_v2.pt'
        if not os.path.exists(pretrained_path):
            print("正在下载预训练模型...")
            os.makedirs('animegan2-pytorch/weights', exist_ok=True)
            try:
                import gdown
                url = 'https://drive.google.com/uc?id=1WK5Mdt6mwlcsqCZMHkCUSDJxN1UyFi0-'
                gdown.download(url, pretrained_path, quiet=False)
            except Exception as e:
                print(f"模型下载失败: {e}")
                print("请手动下载模型文件:")
                print("1. 访问: https://drive.google.com/drive/folders/1Xg1h6DwMDqvXSnoDGbwp37n7fSeHLLjw")
                print("2. 下载 face_paint_512_v2.pt")
                print("3. 将文件放到 animegan2-pytorch/weights/ 目录下")
                return
        
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        model.eval()
        
        # 生成动漫风格图像
        print("正在生成动漫风格头像...")
        with torch.no_grad():
            input_tensor = to_tensor(image).unsqueeze(0) * 2 - 1
            input_tensor = input_tensor.to(device)
            output_tensor = model(input_tensor)
            output_image = to_pil_image((output_tensor.squeeze(0) * 0.5 + 0.5).cpu())
        
        # 转换回OpenCV格式并保存
        output_array = np.array(output_image)
        output_array = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite('anime_style.jpg', output_array)
        
        # 显示结果
        cv2.imshow('Original', cv2.imread('original.jpg'))
        cv2.imshow('Anime Style', cv2.imread('anime_style.jpg'))
        print("结果已保存为 original.jpg 和 anime_style.jpg")
        print("按任意键退出")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    main() 
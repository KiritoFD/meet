import cv2
from avatar_generator.core import AvatarGenerator

def main():
    # 初始化生成器
    generator = AvatarGenerator()
    
    # 读取用户照片
    user_photo = cv2.imread('user_photo.jpg')
    
    # 生成虚拟形象
    avatar = generator.generate_avatar(user_photo)
    
    # 实时预览
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 检测面部特征点
        features = generator.extract_features(frame)
        
        # 实时更新虚拟形象
        updated_avatar = generator.real_time_control(avatar, features)
        
        # 显示结果
        cv2.imshow('Virtual Avatar', updated_avatar)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 
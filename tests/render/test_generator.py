import cv2
from avatar_generator.generator import AvatarGenerator

def main():
    # 初始化生成器
    generator = AvatarGenerator(model_path='models/animegan_v2.pth')
    
    # 捕捉参考照片
    print("请面对摄像头，保持自然表情")
    print("按空格键拍照，按q退出")
    reference_image = generator.capture_reference()
    
    if reference_image is None:
        print("未能捕捉到合适的参考照片")
        return
    
    # 生成虚拟形象
    print("正在生成虚拟形象...")
    avatar = generator.generate_avatar(reference_image)
    
    if avatar is not None:
        # 显示结果
        cv2.imshow('Original', reference_image)
        cv2.imshow('Anime Style', avatar)
        
        # 保存结果
        cv2.imwrite('reference.jpg', reference_image)
        cv2.imwrite('avatar.jpg', avatar)
        
        print("结果已保存为 reference.jpg 和 avatar.jpg")
        print("按任意键退出")
        cv2.waitKey(0)
    else:
        print("生成失败")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
from avatar_generator.core import AvatarGenerator

def main():
    generator = AvatarGenerator()
    
    # 准备训练数据
    generator.prepare_training_data()

if __name__ == '__main__':
    main() 
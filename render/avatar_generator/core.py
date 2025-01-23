import torch
from avatar_generator.dataset import DatasetManager

class AvatarGenerator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.dataset_manager = DatasetManager()
        
        # ... 其余初始化代码保持不变 ...

    def prepare_training_data(self):
        """准备训练数据"""
        self.dataset_manager.prepare_training_data()

    # ... 其余代码保持不变 ... 
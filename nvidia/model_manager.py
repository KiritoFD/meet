import os
import logging
import torch
import numpy as np
from typing import Dict, Optional

# Handle imports in a way that works for both package imports and direct script execution
try:
    from nvidia.model_wrapper import NVIDIAModelWrapper  # Try relative import first
except ImportError:
    from .model_wrapper import NVIDIAModelWrapper  # Try package relative import
from pose.types import PoseData

logger = logging.getLogger(__name__)

__all__ = ['NVIDIAModelManager']

class NVIDIAModelManager:
    """NVIDIA模型管理器"""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = NVIDIAModelManager()
        return cls._instance
    
    def __init__(self):
        """初始化模型管理器"""
        self.model = None
        self.is_initialized = False
        self.model_path = None
        self.config_path = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def initialize(self, config_path=None, checkpoint_path=None):
        """初始化NVIDIA模型
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认路径
            checkpoint_path: 检查点文件路径，如果为None则不加载检查点
            
        Returns:
            初始化是否成功
        """
        try:
            # 设置默认路径
            if config_path is None:
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                config_path = os.path.join(project_root, 'nvidia', 'vox-256.yaml')
                
            if checkpoint_path is None:
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                checkpoint_path = os.path.join(project_root, 'models', 'nvidia_animation.pth')
                
            # 检查模型文件是否存在
            if not os.path.exists(checkpoint_path):
                logger.error(f"模型文件不存在: {checkpoint_path}")
                return False
            
            # 初始化模型
            logger.info("正在初始化NVIDIA模型...")
            self.model = NVIDIAModelWrapper(config_path, checkpoint_path)
            self.is_initialized = True
            self.model_path = checkpoint_path
            self.config_path = config_path
            
            logger.info(f"NVIDIA模型初始化完成，使用设备: {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"初始化NVIDIA模型失败: {str(e)}")
            self.is_initialized = False
            return False
    
    def animate(self, source_image: np.ndarray, pose_data: PoseData) -> Optional[np.ndarray]:
        """根据姿态数据生成动画帧
        
        Args:
            source_image: 源图像 (初始帧)
            pose_data: 姿态数据
            
        Returns:
            生成的图像，如果出错则返回None
        """
        if not self.is_initialized or self.model is None:
            logger.warning("NVIDIA模型未初始化，无法生成动画")
            return None
            
        if source_image is None:
            logger.warning("源图像为空，无法生成动画")
            return None
            
        if pose_data is None or len(pose_data.keypoints) == 0:
            logger.warning("姿态数据为空，无法生成动画")
            return None
            
        try:
            # 生成动画
            result = self.model.animate(source_image, pose_data)
            return result
            
        except Exception as e:
            logger.error(f"生成动画失败: {str(e)}")
            return None
    
    def set_reference_frame(self, frame: np.ndarray) -> bool:
        """设置参考帧并提取关键点
        
        Args:
            frame: 参考帧图像
            
        Returns:
            操作是否成功
        """
        if not self.is_initialized or self.model is None:
            logger.warning("NVIDIA模型未初始化，无法设置参考帧")
            return False
            
        try:
            # 提取并保存源图像关键点
            success = self.model.extract_and_save_source_keypoints(frame)
            return success
        except Exception as e:
            logger.error(f"设置参考帧失败: {str(e)}")
            return False
            
    def get_status(self) -> Dict:
        """获取模型状态"""
        return {
            'initialized': self.is_initialized,
            'device': str(self.device),
            'model_path': self.model_path,
            'config_path': self.config_path,
            'has_source_keypoints': hasattr(self.model, '_source_kp') and self.model._source_kp is not None if self.model else False
        }

import cv2
import logging
import time
import os
import threading
from config.settings import CAMERA_CONFIG
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)

class CameraManager:
    """摄像头管理器"""
    
    def __init__(self, config: Dict = None):
        """初始化摄像头管理器
        
        Args:
            config: 摄像头配置字典，可包含以下键:
                - device_id: 设备ID (默认0)
                - width: 图像宽度
                - height: 图像高度
                - fps: 目标帧率
        """
        # 初始化属性
        self.camera = None
        self.is_running = False
        self.frame_count = 0
        self.start_time = time.time()
        
        # 从配置加载参数
        self.config = config or {}
        self.device_id = self.config.get('device_id', 0)
        self.width = self.config.get('width', 640)
        self.height = self.config.get('height', 480)
        self.fps = self.config.get('fps', 30)
        
        # 添加 frame_rate 属性
        self.frame_rate = config.get('frame_rate', 30) if config else 30
        
        # 预先设置环境变量
        os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'  # 禁用MSMF
        os.environ['OPENCV_VIDEOIO_PRIORITY_DSHOW'] = '1'  # 优先DirectShow
        
        # 使用CAMERA_CONFIG中的参数
        self._camera_params = CAMERA_CONFIG['params']
        self._api_preference = CAMERA_CONFIG['api_preference']
        
    def _init_camera(self):
        """初始化摄像头设备"""
        try:
            # 使用配置的设备ID和API
            self.camera = cv2.VideoCapture(self.device_id, self._api_preference)
            if not self.camera.isOpened():
                return False
                
            # 优先设置禁用弹窗
            self.camera.set(cv2.CAP_PROP_SETTINGS, 0)
            
            # 设置其他参数
            for prop, value in self._camera_params.items():
                if prop != cv2.CAP_PROP_SETTINGS:  # 跳过已设置的禁用弹窗参数
                    self.camera.set(prop, value)
                
            # 快速测试读取
            ret, _ = self.camera.read()
            if ret:
                # 重置帧率计算
                self.frame_count = 0
                self.start_time = time.time()
            return ret
            
        except Exception as e:
            logger.error(f"摄像头初始化失败: {e}")
            return False
            
    def start(self) -> bool:
        """启动摄像头
        
        Returns:
            bool: 是否成功启动
        """
        try:
            if self.is_running:
                return True
                
            # 打开摄像头
            self.camera = cv2.VideoCapture(self.device_id)
            if not self.camera.isOpened():
                raise RuntimeError("无法打开摄像头")
                
            # 设置参数
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
            
            # 重置计数器
            self.frame_count = 0
            self.start_time = time.time()
            self.is_running = True
            
            logger.info("摄像头已启动")
            return True
            
        except Exception as e:
            logger.error(f"启动摄像头失败: {e}")
            self.is_running = False
            return False
            
    def stop(self) -> bool:
        """停止摄像头
        
        Returns:
            bool: 是否成功停止
        """
        try:
            if not self.is_running:
                return True
                
            if self.camera:
                self.camera.release()
                
            self.camera = None
            self.is_running = False
            logger.info("摄像头已停止")
            return True
            
        except Exception as e:
            logger.error(f"停止摄像头失败: {e}")
            return False
            
    def read_frame(self) -> Optional[cv2.Mat]:
        """读取一帧图像
        
        Returns:
            Optional[cv2.Mat]: 图像帧，失败返回None
        """
        if not self.is_running or not self.camera:
            return None
            
        try:
            ret, frame = self.camera.read()
            if not ret:
                return None
                
            self.frame_count += 1
            return frame
            
        except Exception as e:
            logger.error(f"读取帧失败: {e}")
            return None
            
    @property
    def current_fps(self) -> float:
        """获取当前帧率"""
        if not self.is_running:
            return 0.0
            
        elapsed = time.time() - self.start_time
        return self.frame_count / max(elapsed, 0.001)
        
    def release(self):
        """释放资源"""
        self.stop()
        
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'camera'):
            self.release()
        
    def get_settings(self) -> Dict:
        """获取当前相机设置"""
        if not self.camera:
            return {
                "width": self.width,
                "height": self.height,
                "fps": self.frame_rate,  # 使用 frame_rate 而非 current_fps
                "exposure": None,
                "brightness": None,
                "contrast": None
            }
            
        return {
            "width": int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.frame_rate,  # 使用 frame_rate 而非 current_fps
            "exposure": self.camera.get(cv2.CAP_PROP_EXPOSURE),
            "brightness": self.camera.get(cv2.CAP_PROP_BRIGHTNESS),
            "contrast": self.camera.get(cv2.CAP_PROP_CONTRAST)
        }
        
    def update_settings(self, settings: Dict) -> bool:
        """更新相机设置
        
        Args:
            settings: 设置字典
            
        Returns:
            bool: 是否成功
        """
        if not self.camera:
            return False
        
        try:
            for key, value in settings.items():
                if key == 'brightness':
                    self.camera.set(cv2.CAP_PROP_BRIGHTNESS, value)
                elif key == 'contrast':
                    self.camera.set(cv2.CAP_PROP_CONTRAST, value)
                elif key == 'exposure':
                    self.camera.set(cv2.CAP_PROP_EXPOSURE, value)
                elif key == 'gain':
                    self.camera.set(cv2.CAP_PROP_GAIN, value)
                elif key == 'width':
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, value)
                elif key == 'height':
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, value)
                elif key == 'fps':
                    self.frame_rate = value
                    if self.camera and self.is_running:
                        self.camera.set(cv2.CAP_PROP_FPS, self.frame_rate)
                
            return True
            
        except Exception as e:
            logger.error(f"更新相机设置失败: {e}")
            return False
            
    def reset_settings(self) -> bool:
        """重置相机设置"""
        if not self.camera:
            return False
        
        try:
            self.camera.set(cv2.CAP_PROP_BRIGHTNESS, 50)
            self.camera.set(cv2.CAP_PROP_CONTRAST, 50)
            self.camera.set(cv2.CAP_PROP_EXPOSURE, -6)
            self.camera.set(cv2.CAP_PROP_GAIN, 0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return True
            
        except Exception as e:
            logger.error(f"重置相机设置失败: {e}")
            return False

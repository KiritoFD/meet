import cv2
import time
from utils.logger import logger
from config.settings import CAMERA_CONFIG

class CameraManager:
    def __init__(self):
        self.camera = None
        self.current_frame = None
        self.current_pose = None

    def initialize(self):
        """初始化摄像头"""
        try:
            if self.camera is not None:
                self.camera.release()
                self.camera = None
            
            time.sleep(0.1)  # 等待摄像头完全释放
            
            self.camera = cv2.VideoCapture(CAMERA_CONFIG['index'])
            if not self.camera.isOpened():
                logger.error("无法打开摄像头")
                return False
            
            # 先读取一帧测试
            success, frame = self.camera.read()
            if not success:
                logger.error("无法读取摄像头帧")
                return False
            
            # 设置摄像头参数
            self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG['width'])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG['height'])
            self.camera.set(cv2.CAP_PROP_FPS, CAMERA_CONFIG['fps'])
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_CONFIG['buffer_size'])
            self.camera.set(cv2.CAP_PROP_CONVERT_RGB, True)
            
            # 再次测试读取
            success, frame = self.camera.read()
            if success and frame is not None:
                logger.info(f"摄像头初始化成功: {frame.shape}")
                return True
                
            logger.error("摄像头初始化失败：无法读取有效帧")
            return False
            
        except Exception as e:
            logger.error(f"摄像头初始化失败: {str(e)}")
            return False

    def read_frame(self):
        """读取一帧"""
        if self.camera is None or not self.camera.isOpened():
            return None
        success, frame = self.camera.read()
        if success:
            self.current_frame = frame
            return frame
        return None

    def release(self):
        """释放摄像头"""
        if self.camera is not None:
            logger.debug("开始释放摄像头...")
            self.camera.release()
            self.camera = None
            logger.info("摄像头已释放")
        else:
            logger.debug("摄像头已经是空的")

    def is_active(self):
        """检查摄像头状态"""
        if self.camera is None:
            return False
        try:
            return self.camera.isOpened() and self.camera.read()[0]
        except Exception as e:
            logger.error(f"检查摄像头状态失败: {str(e)}")
            return False 
import cv2
import logging
import time
import os
import threading
from config.settings import CAMERA_CONFIG

logger = logging.getLogger(__name__)

class CameraManager:
    def __init__(self):
        """初始化摄像头管理器"""
        self.camera = None
        self._is_running = False
        self._init_lock = threading.Lock()
        
        # 预先设置环境变量
        os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'   # 禁用MSMF
        os.environ['OPENCV_VIDEOIO_PRIORITY_DSHOW'] = '1'  # 优先DirectShow
        
        # 基本参数设置
        self._camera_params = {
            cv2.CAP_PROP_FRAME_WIDTH: 640,     # 基础分辨率
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 30,              # 标准帧率
            cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*'MJPG'),  # MJPG格式
            cv2.CAP_PROP_EXPOSURE: 0,          # 自动曝光
            cv2.CAP_PROP_BRIGHTNESS: 128,      # 标准亮度
            cv2.CAP_PROP_CONTRAST: 128,        # 标准对比度
            cv2.CAP_PROP_SATURATION: 128,      # 标准饱和度
            cv2.CAP_PROP_AUTO_WB: 1,           # 自动白平衡
            cv2.CAP_PROP_AUTOFOCUS: 0          # 禁用自动对焦
        }
        
    def _init_camera(self):
        """初始化摄像头设备"""
        try:
            # 快速初始化
            self.camera = cv2.VideoCapture(CAMERA_CONFIG['device_id'], cv2.CAP_DSHOW)
            if not self.camera.isOpened():
                return False
            
            # 优先设置基本参数
            success = True
            critical_params = {
                cv2.CAP_PROP_FRAME_WIDTH: self._camera_params[cv2.CAP_PROP_FRAME_WIDTH],
                cv2.CAP_PROP_FRAME_HEIGHT: self._camera_params[cv2.CAP_PROP_FRAME_HEIGHT],
                cv2.CAP_PROP_FPS: self._camera_params[cv2.CAP_PROP_FPS],
                cv2.CAP_PROP_FOURCC: self._camera_params[cv2.CAP_PROP_FOURCC]
            }
            
            # 设置关键参数
            for prop, value in critical_params.items():
                if not self.camera.set(prop, value):
                    logger.warning(f"设置参数失败: {prop}={value}")
                    success = False
            
            # 快速测试读取
            ret, frame = self.camera.read()
            if not ret or frame is None:
                return False
                
            # 后台设置其他参数
            threading.Thread(target=self._set_quality_params, daemon=True).start()
            
            return success
            
        except Exception as e:
            logger.error(f"摄像头初始化失败: {e}")
            return False
            
    def _set_quality_params(self):
        """后台设置图像质量参数"""
        try:
            time.sleep(0.1)  # 等待摄像头稳定
            quality_params = {
                cv2.CAP_PROP_BRIGHTNESS: self._camera_params[cv2.CAP_PROP_BRIGHTNESS],
                cv2.CAP_PROP_CONTRAST: self._camera_params[cv2.CAP_PROP_CONTRAST],
                cv2.CAP_PROP_SATURATION: self._camera_params[cv2.CAP_PROP_SATURATION],
                cv2.CAP_PROP_AUTO_WB: self._camera_params[cv2.CAP_PROP_AUTO_WB],
                cv2.CAP_PROP_EXPOSURE: self._camera_params[cv2.CAP_PROP_EXPOSURE],
                cv2.CAP_PROP_AUTOFOCUS: self._camera_params[cv2.CAP_PROP_AUTOFOCUS]
            }
            
            for prop, value in quality_params.items():
                try:
                    if not self.camera.set(prop, value):
                        logger.debug(f"设置图像质量参数失败: {prop}={value}")
                except:
                    pass  # 忽略设置失败
                    
        except Exception as e:
            logger.debug(f"设置图像质量参数错误: {e}")
    
    def start(self):
        """启动摄像头"""
        with self._init_lock:
            if self._is_running:
                return True
                
            try:
                start_time = time.time()
                logger.info("开始初始化摄像头...")
                
                # 设置较短的超时时间
                timeout = 2.0  # 2秒超时
                while time.time() - start_time < timeout:
                    if self._init_camera():
                        self._is_running = True
                        elapsed = time.time() - start_time
                        logger.info(f"摄像头初始化完成，耗时: {elapsed:.2f}秒")
                        return True
                    
                    # 快速重试
                    if self.camera:
                        self.camera.release()
                        self.camera = None
                    time.sleep(0.1)
                    
                logger.error("摄像头启动超时")
                return False
                
            except Exception as e:
                logger.error(f"摄像头启动失败: {e}")
                self.stop()
                return False
    
    def stop(self):
        """停止摄像头"""
        if self.camera:
            self.camera.release()
            self.camera = None
        self._is_running = False
    
    def read(self):
        """读取一帧"""
        if not self._is_running or not self.camera:
            return False, None
            
        try:
            return self.camera.read()
        except Exception as e:
            logger.error(f"读取帧失败: {e}")
            self.stop()
            return False, None
    
    @property
    def is_running(self):
        """检查摄像头是否正在运行"""
        return self._is_running and self.camera is not None

    def __del__(self):
        """析构函数确保资源释放"""
        self.stop()
        
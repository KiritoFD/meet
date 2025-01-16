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
        self._init_lock = threading.Lock()  # 添加初始化锁
        
        # 帧率计算相关属性
        self.frame_count = 0
        self.fps_time = time.time()
        self.current_fps = 0
        
        # 预先设置环境变量
        os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'  # 禁用MSMF
        os.environ['OPENCV_VIDEOIO_PRIORITY_DSHOW'] = '1'  # 优先DirectShow
        
        # 关键参数设置
        self._camera_params = {
            cv2.CAP_PROP_SETTINGS: 0,         # 禁用设置弹窗
            cv2.CAP_PROP_EXPOSURE: -6,        # 自动曝光
            cv2.CAP_PROP_AUTOFOCUS: 0,        # 禁用自动对焦
            cv2.CAP_PROP_BUFFERSIZE: 1,       # 最小缓冲
            cv2.CAP_PROP_FPS: 30,             # 帧率
            cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*'MJPG'),  # MJPG格式
            cv2.CAP_PROP_SETTINGS: 0          # 禁用所有弹窗
        }
        
    def _init_camera(self):
        """初始化摄像头设备"""
        try:
            # 在创建摄像头对象时就设置禁用弹窗
            self.camera = cv2.VideoCapture(CAMERA_CONFIG['device_id'], cv2.CAP_DSHOW)
            if not self.camera.isOpened():
                return False
                
            # 优先设置禁用弹窗
            self.camera.set(cv2.CAP_PROP_SETTINGS, 0)
            
            # 然后设置其他参数
            for prop, value in self._camera_params.items():
                if prop != cv2.CAP_PROP_SETTINGS:  # 跳过已设置的禁用弹窗参数
                    self.camera.set(prop, value)
                
            # 快速测试读取
            ret, _ = self.camera.read()
            if ret:
                # 重置帧率计算
                self.frame_count = 0
                self.fps_time = time.time()
                self.current_fps = 0
            return ret
            
        except Exception as e:
            logger.error(f"摄像头初始化失败: {e}")
            return False
            
    def start(self):
        """启动摄像头"""
        # 使用锁防止并发初始化
        with self._init_lock:
            if self._is_running:
                return True
                
            try:
                start_time = time.time()
                
                # 设置超时
                timeout = 3
                while time.time() - start_time < timeout:
                    if self._init_camera():
                        self._is_running = True
                        end_time = time.time()
                        logger.info(f"摄像头启动成功，耗时: {(end_time-start_time):.2f}秒")
                        return True
                        
                    # 如果失败，快速重试
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
        # 重置帧率计数
        self.frame_count = 0
        self.current_fps = 0
        
    def read(self):
        """读取一帧并计算帧率"""
        if not self._is_running or not self.camera:
            return False, None
            
        try:
            ret, frame = self.camera.read()
            if ret:
                # 处理帧率计算
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.fps_time >= 1.0:
                    self.current_fps = self.frame_count
                    self.frame_count = 0
                    self.fps_time = current_time
                    logger.debug(f"当前FPS: {self.current_fps}")
            return ret, frame
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
        
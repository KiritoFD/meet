import cv2
import time
import logging
import threading
import numpy as np
from typing import Dict, Optional, List, Generator, Tuple

from pose.detector import PoseDetector
from pose.pose_deformer import PoseDeformer
from pose.binding import PoseBinding
from camera.manager import CameraManager
from pose.drawer import PoseDrawer
from pose.smoother import FrameSmoother
from nvidia.model_manager import NVIDIAModelManager

logger = logging.getLogger(__name__)

class StreamManager:
    """视频流管理器：处理摄像头捕获、姿态处理、流式传输"""
    
    def __init__(self, camera_config: Dict):
        """初始化流管理器
        
        Args:
            camera_config: 相机配置参数
        """
        self.camera = CameraManager(camera_config)
        self.detector = PoseDetector()
        self.drawer = PoseDrawer()
        self.smoother = FrameSmoother()
        self.binding = None  # 会在捕获参考帧后初始化
        self.deformer = None  # 会在捕获参考帧后初始化
        
        # 初始化NVIDIA模型管理器
        self.nvidia_model = NVIDIAModelManager.get_instance()
        self.use_nvidia_model = False
        
        self.reference_frame = None
        self.reference_regions = None
        
        self.is_running = False
        self.thread = None
        self.lock = threading.RLock()
        
        # 性能统计
        self.frame_count = 0
        self.start_time = 0
        self.current_fps = 0
        self.processing_times = []  # 存储最近几帧处理时间
        
        # 流配置
        self.quality = 85  # JPEG 质量
        self.max_width = 640  # 最大宽度
        self.downsample = False  # 是否下采样
        
    def start(self) -> bool:
        """启动流处理"""
        if self.is_running:
            return True
            
        if not self.camera.start():
            logger.error("无法启动摄像头")
            return False
            
        self.is_running = True
        self.frame_count = 0
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        
        logger.info("流处理已启动")
        return True
        
    def stop(self) -> None:
        """停止流处理"""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.camera.stop()
        logger.info("流处理已停止")
        
    def set_reference(self, frame: np.ndarray, pose_data) -> bool:
        """设置参考帧和区域绑定
        
        Args:
            frame: 参考图像帧
            pose_data: 对应的姿态数据
            
        Returns:
            设置是否成功
        """
        try:
            with self.lock:
                self.reference_frame = frame.copy()
                
                # 初始化姿态绑定器
                if self.binding is None:
                    self.binding = PoseBinding()
                    
                # 创建区域绑定
                self.reference_regions = self.binding.create_binding(frame, pose_data)
                
                # 初始化变形器
                if self.deformer is None:
                    self.deformer = PoseDeformer()
                
                # 初始化NVIDIA模型（如果尚未初始化）
                if not self.nvidia_model.is_initialized:
                    self.nvidia_model.initialize()
                    # 只有在成功初始化模型后才启用
                    self.use_nvidia_model = self.nvidia_model.is_initialized
                
                return True
        except Exception as e:
            logger.error(f"设置参考帧失败: {str(e)}")
            return False
            
    def generate_frames(self) -> Generator[bytes, None, None]:
        """生成视频帧流
        
        Yields:
            编码后的图像帧用于HTTP流式传输
        """
        while self.is_running:
            start_time = time.time()
            
            # 获取处理后的帧
            processed_frame = self._get_processed_frame()
            
            if processed_frame is not None:
                # 根据需要调整尺寸
                if self.downsample and processed_frame.shape[1] > self.max_width:
                    scale = self.max_width / processed_frame.shape[1]
                    new_height = int(processed_frame.shape[0] * scale)
                    processed_frame = cv2.resize(processed_frame, (self.max_width, new_height))
                
                # 帧率显示
                self._update_fps()
                cv2.putText(
                    processed_frame, 
                    f"FPS: {self.current_fps:.1f}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2
                )
                
                # 编码帧
                try:
                    ret, buffer = cv2.imencode(
                        '.jpg', 
                        processed_frame, 
                        [cv2.IMWRITE_JPEG_QUALITY, self.quality]
                    )
                    
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception as e:
                    logger.error(f"编码帧失败: {str(e)}")
            
            # 控制帧率
            elapsed = time.time() - start_time
            if elapsed < 1/30:  # 目标30fps
                time.sleep(1/30 - elapsed)
    
    def _process_loop(self) -> None:
        """后台处理循环"""
        while self.is_running:
            try:
                self._process_frame()
            except Exception as e:
                logger.error(f"处理帧时出错: {str(e)}")
                time.sleep(0.01)  # 防止CPU过度使用
    
    def _process_frame(self) -> None:
        """处理单帧图像"""
        if not self.camera.is_running:
            return
            
        # 读取帧
        frame = self.camera.read_frame()
        if frame is None:
            return
            
        self.frame_count += 1
            
    def _get_processed_frame(self) -> Optional[np.ndarray]:
        """获取处理后的当前帧"""
        if not self.camera.is_running:
            return None
            
        # 获取原始帧
        frame = self.camera.read_frame()
        if frame is None:
            return None
            
        # 转换颜色空间
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 检测姿态
        pose_data = self.detector.detect(frame)
        
        # 绘制结果
        output_frame = frame.copy()
        
        # 如果存在参考帧和姿态数据，应用变形或NVIDIA模型
        if (self.reference_frame is not None and pose_data is not None):
            
            try:
                # 应用平滑处理
                if self.smoother:
                    pose_data = self.smoother.smooth(pose_data)
                
                # 根据配置选择使用NVIDIA模型或传统变形
                if self.use_nvidia_model:
                    # 使用NVIDIA模型生成动画
                    animated_frame = self.nvidia_model.animate(self.reference_frame, pose_data)
                    if animated_frame is not None:
                        output_frame = animated_frame
                else:
                    # 使用传统变形方法
                    if self.reference_regions is not None and self.deformer is not None:
                        output_frame = self.deformer.deform_frame(
                            self.reference_frame,
                            self.reference_regions,
                            pose_data
                        )
                
            except Exception as e:
                logger.error(f"处理失败: {str(e)}")
        
        # 绘制姿态关键点
        if pose_data is not None:
            output_frame = self.drawer.draw_pose(output_frame, pose_data)
        
        return output_frame
        
    def _update_fps(self) -> None:
        """更新FPS计算"""
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.current_fps = self.frame_count / elapsed
            
            # 每5秒重置计数器，避免长时间运行导致平均值不准确
            if elapsed > 5:
                self.frame_count = 0
                self.start_time = time.time()
                
    def get_status(self) -> Dict:
        """获取流状态信息"""
        status = {
            'running': self.is_running,
            'fps': self.current_fps,
            'frame_count': self.frame_count,
            'camera_status': self.camera.is_running,
            'has_reference': self.reference_frame is not None,
            'resolution': {
                'width': self.camera.width,
                'height': self.camera.height
            } if self.camera.is_running else None
        }
        
        # 添加NVIDIA模型状态
        status['nvidia_model'] = {
            'enabled': self.use_nvidia_model,
            'initialized': self.nvidia_model.is_initialized
        }
        
        return status
    
    def toggle_nvidia_model(self, enable: bool) -> bool:
        """切换是否使用NVIDIA模型
        
        Args:
            enable: 是否启用NVIDIA模型
            
        Returns:
            操作是否成功
        """
        try:
            if enable and not self.nvidia_model.is_initialized:
                success = self.nvidia_model.initialize()
                if not success:
                    return False
            
            self.use_nvidia_model = enable and self.nvidia_model.is_initialized
            return True
        except Exception as e:
            logger.error(f"切换NVIDIA模型失败: {str(e)}")
            return False

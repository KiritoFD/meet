import cv2
import time
import threading
import logging
import numpy as np
from typing import Dict, Optional, List, Tuple, Any
from queue import Queue, Empty, Full

from pose.types import PoseData
from nvidia.model_manager import NVIDIAModelManager
from nvidia.keypoint_compressor import KeypointCompressor
from nvidia.network_simulator import NetworkSimulator

logger = logging.getLogger(__name__)

class KeypointReceiver:
    """关键点接收器，从网络模拟器接收关键点数据并生成视频流"""
    
    def __init__(self, reference_frame: np.ndarray = None):
        """初始化关键点接收器
        
        Args:
            reference_frame: 参考帧图像，用于生成动画
        """
        self.reference_frame = reference_frame
        self.nvidia_model = NVIDIAModelManager.get_instance()
        self.keypoint_compressor = KeypointCompressor(precision=2)
        
        # 用于存储接收到的关键点数据
        self.keypoint_queue = Queue(maxsize=30)  # 存储最多30帧数据
        
        # 生成的帧缓冲区
        self.frame_buffer = Queue(maxsize=10)
        
        # 处理状态
        self.is_running = False
        self.receive_thread = None
        self.process_thread = None
        self.lock = threading.RLock()
        
        # 性能统计
        self.stats = {
            'received_packets': 0,
            'processed_frames': 0,
            'dropped_frames': 0,
            'last_timestamp': 0,
            'latency_ms': 0,
            'current_fps': 0,
            'start_time': 0
        }
        
        # 控制参数
        self.target_fps = 30
        
    def set_reference_frame(self, frame: np.ndarray) -> bool:
        """设置参考帧
        
        Args:
            frame: 参考帧图像
            
        Returns:
            设置是否成功
        """
        if frame is None:
            return False
            
        with self.lock:
            self.reference_frame = frame.copy()
            
            # 初始化NVIDIA模型（如果尚未初始化）
            if not self.nvidia_model.is_initialized:
                self.nvidia_model.initialize()
                
            # 为NVIDIA模型设置参考帧
            if self.nvidia_model.is_initialized:
                self.nvidia_model.set_reference_frame(frame)
                
            logger.info("参考帧已设置")
            return True
            
    def start(self) -> bool:
        """启动接收器处理"""
        if self.is_running:
            return True
            
        if self.reference_frame is None:
            logger.error("未设置参考帧，无法启动接收器")
            return False
            
        if not self.nvidia_model.is_initialized:
            logger.error("NVIDIA模型未初始化，无法启动接收器")
            return False
            
        self.is_running = True
        self.stats['start_time'] = time.time()
        self.stats['received_packets'] = 0
        self.stats['processed_frames'] = 0
        self.stats['dropped_frames'] = 0
        
        # 启动接收线程
        self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.receive_thread.start()
        
        # 启动处理线程
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        
        logger.info("关键点接收器已启动")
        return True
        
    def stop(self) -> None:
        """停止接收器处理"""
        self.is_running = False
        
        # 等待线程结束
        if self.receive_thread and self.receive_thread.is_alive():
            self.receive_thread.join(timeout=2.0)
            
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=2.0)
            
        # 清空队列
        while not self.keypoint_queue.empty():
            try:
                self.keypoint_queue.get_nowait()
            except Empty:
                break
                
        while not self.frame_buffer.empty():
            try:
                self.frame_buffer.get_nowait()
            except Empty:
                break
                
        logger.info("关键点接收器已停止")
        
    def receive_keypoint_data(self, data: Dict, simulated_network: bool = True) -> bool:
        """接收关键点数据
        
        Args:
            data: 关键点数据字典或序列化的数据
            simulated_network: 是否使用网络模拟器模拟网络传输
            
        Returns:
            接收是否成功
        """
        try:
            # 记录接收时间戳
            receive_time = time.time()
            
            # 1. 如果是序列化数据，先反序列化
            if isinstance(data, str):
                data = self.keypoint_compressor.deserialize_from_transmission(data)
                
            # 2. 解压缩关键点数据
            pose_data = self.keypoint_compressor.decompress_pose_data(data)
            if pose_data is None:
                logger.warning("解压缩关键点数据失败")
                return False
                
            # 3. 计算延迟
            if pose_data.timestamp > 0:
                latency = (receive_time - pose_data.timestamp) * 1000  # 毫秒
                self.stats['latency_ms'] = latency
                
            # 4. 存入队列
            try:
                # 如果队列满了，移除最旧的数据
                if self.keypoint_queue.full():
                    try:
                        self.keypoint_queue.get_nowait()
                        self.stats['dropped_frames'] += 1
                    except Empty:
                        pass
                
                # 添加到队列
                self.keypoint_queue.put_nowait({
                    'pose_data': pose_data,
                    'receive_time': receive_time
                })
                
                # 更新统计信息
                self.stats['received_packets'] += 1
                self.stats['last_timestamp'] = receive_time
                
                return True
                
            except Full:
                self.stats['dropped_frames'] += 1
                logger.warning("关键点队列已满，丢弃数据")
                return False
                
        except Exception as e:
            logger.error(f"接收关键点数据失败: {str(e)}")
            return False
    
    def _receive_loop(self) -> None:
        """接收循环，模拟从网络接收数据"""
        # 此方法仅用于测试，实际应用中应通过网络接口接收数据
        pass
    
    def _process_loop(self) -> None:
        """处理循环，生成动画帧"""
        target_interval = 1.0 / self.target_fps
        last_frame_time = 0
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # 控制帧率
                elapsed = current_time - last_frame_time
                if elapsed < target_interval:
                    time.sleep(target_interval - elapsed)
                    continue
                    
                last_frame_time = current_time
                
                # 获取最新的关键点数据
                keypoint_data = None
                while not self.keypoint_queue.empty():
                    try:
                        keypoint_data = self.keypoint_queue.get_nowait()
                    except Empty:
                        break
                
                # 如果没有数据，跳过此帧
                if keypoint_data is None:
                    continue
                    
                # 生成动画帧
                pose_data = keypoint_data['pose_data']
                frame = self._generate_frame(pose_data)
                
                if frame is not None:
                    # 计算帧率
                    elapsed = time.time() - self.stats['start_time']
                    if elapsed > 0:
                        self.stats['current_fps'] = self.stats['processed_frames'] / elapsed
                        
                        # 每5秒重置计数器，以获得更准确的当前帧率
                        if elapsed > 5:
                            self.stats['processed_frames'] = 0
                            self.stats['start_time'] = time.time()
                    
                    # 添加到帧缓冲区
                    try:
                        # 如果缓冲区已满，移除最旧的帧
                        if self.frame_buffer.full():
                            try:
                                self.frame_buffer.get_nowait()
                            except Empty:
                                pass
                                
                        self.frame_buffer.put_nowait(frame)
                        self.stats['processed_frames'] += 1
                        
                    except Full:
                        logger.warning("帧缓冲区已满，丢弃帧")
                
            except Exception as e:
                logger.error(f"处理关键点数据失败: {str(e)}")
                time.sleep(0.01)  # 避免CPU占用过高
    
    def _generate_frame(self, pose_data: PoseData) -> Optional[np.ndarray]:
        """根据关键点数据生成动画帧
        
        Args:
            pose_data: 姿态数据
            
        Returns:
            生成的帧，如果失败返回None
        """
        if self.reference_frame is None:
            return None
            
        try:
            # 使用NVIDIA模型生成动画帧
            frame = self.nvidia_model.animate(self.reference_frame, pose_data)
            
            if frame is not None:
                # 添加调试信息
                cv2.putText(
                    frame,
                    f"FPS: {self.stats['current_fps']:.1f} | Latency: {self.stats['latency_ms']:.1f}ms",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                # 添加队列状态
                cv2.putText(
                    frame,
                    f"Queue: {self.keypoint_queue.qsize()}/{self.keypoint_queue.maxsize} | "
                    f"Buffer: {self.frame_buffer.qsize()}/{self.frame_buffer.maxsize}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
            return frame
            
        except Exception as e:
            logger.error(f"生成动画帧失败: {str(e)}")
            return None
    
    def get_frame(self) -> Optional[np.ndarray]:
        """获取最新的生成帧
        
        Returns:
            最新的生成帧，如果没有可用帧则返回None
        """
        try:
            # 尝试从缓冲区获取最新帧
            return self.frame_buffer.get_nowait()
        except Empty:
            return None
    
    def get_status(self) -> Dict:
        """获取接收器状态
        
        Returns:
            状态信息字典
        """
        return {
            'running': self.is_running,
            'has_reference': self.reference_frame is not None,
            'received_packets': self.stats['received_packets'],
            'processed_frames': self.stats['processed_frames'],
            'dropped_frames': self.stats['dropped_frames'],
            'current_fps': self.stats['current_fps'],
            'latency_ms': self.stats['latency_ms'],
            'queue_size': self.keypoint_queue.qsize(),
            'buffer_size': self.frame_buffer.qsize()
        }

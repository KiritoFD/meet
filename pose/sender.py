import logging
import time
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class PoseSender:
    """
    负责通过Socket.IO发送姿势数据给客户端的类
    """
    def __init__(self):
        self.socketio = None
        self.connected = False
        self.last_sent_time = 0
        self.send_interval = 1.0 / 30  # 默认30fps
        self.stats = {
            'frames_sent': 0,
            'bytes_sent': 0,
            'start_time': time.time()
        }

    def connect(self, socketio_instance):
        """连接到Socket.IO实例"""
        self.socketio = socketio_instance
        self.connected = True
        self.stats['start_time'] = time.time()
        logger.info("PoseSender connected to Socket.IO")
        return True

    def disconnect(self):
        """断开Socket.IO连接"""
        self.connected = False
        logger.info("PoseSender disconnected from Socket.IO")
        return True

    def send_pose_data(self, pose_data: Dict[str, List[Dict[str, float]]]) -> bool:
        """
        发送姿势数据到客户端
        
        Args:
            pose_data: 包含姿势关键点的字典
                {
                    'pose': [{'x': float, 'y': float, 'z': float, 'visibility': float}, ...],
                    'face': [...],
                    'left_hand': [...],
                    'right_hand': [...]
                }
        
        Returns:
            bool: 发送是否成功
        """
        if not self.connected or not self.socketio:
            logger.warning("Cannot send pose data: not connected to Socket.IO")
            return False
            
        # 限制发送频率
        current_time = time.time()
        if current_time - self.last_sent_time < self.send_interval:
            return False
            
        try:
            # 发送数据
            self.socketio.emit('pose_data', pose_data)
            
            # 更新统计信息
            self.stats['frames_sent'] += 1
            # 粗略估计发送的字节数
            estimated_bytes = sum(len(str(point)) for points in pose_data.values() for point in points)
            self.stats['bytes_sent'] += estimated_bytes
            
            self.last_sent_time = current_time
            return True
            
        except Exception as e:
            logger.error(f"Error sending pose data: {str(e)}")
            return False
    
    def set_fps(self, fps: int) -> None:
        """设置发送帧率"""
        if fps <= 0:
            logger.warning(f"Invalid FPS value: {fps}")
            return
            
        self.send_interval = 1.0 / fps
        logger.info(f"PoseSender FPS set to {fps}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取发送统计信息"""
        elapsed_time = time.time() - self.stats['start_time']
        if elapsed_time > 0:
            fps = self.stats['frames_sent'] / elapsed_time
            kbps = (self.stats['bytes_sent'] / 1024) / elapsed_time
        else:
            fps = 0
            kbps = 0
            
        return {
            'connected': self.connected,
            'frames_sent': self.stats['frames_sent'],
            'bytes_sent': self.stats['bytes_sent'],
            'elapsed_time': elapsed_time,
            'average_fps': fps,
            'average_kbps': kbps
        }

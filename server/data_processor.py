import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import zlib
from dataclasses import dataclass, asdict
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataPacket:
    type: str
    data: Dict
    timestamp: float = None
    sequence: int = 0
    compressed: bool = False
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class DataProcessor:
    def __init__(self):
        """初始化数据处理器"""
        self.compression_level = 6  # zlib压缩级别
        self.pose_buffer: List[Dict] = []
        self.audio_buffer: List[bytes] = []
        self.buffer_size = 30  # 缓冲区大小（帧数）
        self.sequence_counter = 0
        self.last_process_time = time.time()
        
    def process_pose_data(self, pose_data: Dict) -> Dict:
        """处理姿态数据"""
        try:
            # 1. 数据验证
            if not self._validate_pose_data(pose_data):
                raise ValueError("无效的姿态数据")
                
            # 2. 数据标准化
            normalized_data = self._normalize_pose_data(pose_data)
            
            # 3. 添加到缓冲区
            self.pose_buffer.append(normalized_data)
            if len(self.pose_buffer) > self.buffer_size:
                self.pose_buffer.pop(0)
                
            # 4. 平滑处理
            smoothed_data = self._smooth_pose_data(normalized_data)
            
            # 5. 计算处理延迟
            current_time = time.time()
            process_delay = current_time - self.last_process_time
            self.last_process_time = current_time
            
            logger.debug(f"姿态数据处理延迟: {process_delay*1000:.2f}ms")
            return smoothed_data
            
        except Exception as e:
            logger.error(f"处理姿态数据失败: {e}")
            return pose_data
        
    def process_audio_data(self, audio_data: bytes) -> bytes:
        """处理音频数据"""
        try:
            # 1. 添加到缓冲区
            self.audio_buffer.append(audio_data)
            if len(self.audio_buffer) > self.buffer_size:
                self.audio_buffer.pop(0)
                
            # TODO: 实现以下功能
            # 1. 音频降噪
            # 2. 回声消除
            # 3. 音量均衡化
            # 4. 编码优化
            
            return audio_data
            
        except Exception as e:
            logger.error(f"处理音频数据失败: {e}")
            return audio_data
        
    def create_packet(self, data_type: str, data: Dict) -> DataPacket:
        """创建数据包"""
        try:
            packet = DataPacket(
                type=data_type,
                data=data,
                sequence=self._get_next_sequence()
            )
            return packet
        except Exception as e:
            logger.error(f"创建数据包失败: {e}")
            raise
        
    def serialize_packet(self, packet: DataPacket) -> bytes:
        """序列化数据包"""
        try:
            # 1. 转换为字典
            packet_dict = asdict(packet)
            
            # 2. 转换为JSON
            json_data = json.dumps(packet_dict)
            
            # 3. 压缩
            compressed = zlib.compress(json_data.encode(), self.compression_level)
            
            return compressed
            
        except Exception as e:
            logger.error(f"序列化数据包失败: {e}")
            raise
        
    def deserialize_packet(self, data: bytes) -> DataPacket:
        """反序列化数据包"""
        try:
            # 1. 解压缩
            decompressed = zlib.decompress(data)
            
            # 2. 解析JSON
            packet_dict = json.loads(decompressed)
            
            # 3. 转换为DataPacket对象
            return DataPacket(**packet_dict)
            
        except Exception as e:
            logger.error(f"反序列化数据包失败: {e}")
            raise
        
    def _validate_pose_data(self, pose_data: Dict) -> bool:
        """验证姿态数据的有效性"""
        try:
            # 检查必要字段
            required_fields = ["landmarks"]
            if not all(field in pose_data for field in required_fields):
                return False
                
            # 检查关键点数据
            landmarks = pose_data["landmarks"]
            if not landmarks or not isinstance(landmarks, list):
                return False
                
            # 检查每个关键点的格式
            for landmark in landmarks:
                required_landmark_fields = ["x", "y", "z", "visibility"]
                if not all(field in landmark for field in required_landmark_fields):
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"验证姿态数据失败: {e}")
            return False
        
    def _normalize_pose_data(self, pose_data: Dict) -> Dict:
        """标准化姿态数据"""
        try:
            landmarks = pose_data["landmarks"]
            
            # 计算边界框
            x_coords = [lm["x"] for lm in landmarks]
            y_coords = [lm["y"] for lm in landmarks]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # 标准化到0-1范围
            for lm in landmarks:
                lm["x"] = (lm["x"] - x_min) / (x_max - x_min) if x_max > x_min else 0
                lm["y"] = (lm["y"] - y_min) / (y_max - y_min) if y_max > y_min else 0
                
            return pose_data
            
        except Exception as e:
            logger.error(f"标准化姿态数据失败: {e}")
            return pose_data
        
    def _smooth_pose_data(self, pose_data: Dict) -> Dict:
        """平滑姿态数据"""
        try:
            if len(self.pose_buffer) < 3:
                return pose_data
                
            # 使用简单的移动平均
            smoothed_landmarks = []
            current_landmarks = pose_data["landmarks"]
            
            for i in range(len(current_landmarks)):
                x_values = [frame["landmarks"][i]["x"] for frame in self.pose_buffer[-3:]]
                y_values = [frame["landmarks"][i]["y"] for frame in self.pose_buffer[-3:]]
                z_values = [frame["landmarks"][i]["z"] for frame in self.pose_buffer[-3:]]
                
                smoothed_landmarks.append({
                    "x": sum(x_values) / len(x_values),
                    "y": sum(y_values) / len(y_values),
                    "z": sum(z_values) / len(z_values),
                    "visibility": current_landmarks[i]["visibility"]
                })
                
            return {"landmarks": smoothed_landmarks}
            
        except Exception as e:
            logger.error(f"平滑姿态数据失败: {e}")
            return pose_data
        
    def _get_next_sequence(self) -> int:
        """获取下一个序列号"""
        self.sequence_counter = (self.sequence_counter + 1) % 65536
        return self.sequence_counter
        
    def calculate_bandwidth(self, packet: bytes) -> float:
        """计算带宽使用（bytes/s）"""
        try:
            # 简单估算：数据包大小 * 每秒发送的包数
            packet_size = len(packet)
            packets_per_second = self.buffer_size
            bandwidth = packet_size * packets_per_second
            
            logger.debug(f"当前带宽使用: {bandwidth/1024:.2f} KB/s")
            return bandwidth
            
        except Exception as e:
            logger.error(f"计算带宽失败: {e}")
            return 0
        
    def optimize_packet(self, packet: bytes, target_size: int) -> bytes:
        """优化数据包大小"""
        try:
            # TODO: 实现以下优化
            # 1. 数据压缩优化
            # 2. 精度调整
            # 3. 关键帧检测
            # 4. 增量更新
            return packet
            
        except Exception as e:
            logger.error(f"优化数据包失败: {e}")
            return packet
        
    def __del__(self):
        """清理资源"""
        # 清空缓冲区
        self.pose_buffer.clear()
        self.audio_buffer.clear() 
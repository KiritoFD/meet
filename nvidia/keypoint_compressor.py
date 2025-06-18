import json
import zlib
import base64
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pose.types import PoseData

class KeypointCompressor:
    """关键点数据压缩器，用于减小传输数据大小"""
    
    def __init__(self, precision: int = 2):
        """初始化压缩器
        
        Args:
            precision: 小数点精度，默认保留2位小数
        """
        self.precision = precision
        
    def compress_pose_data(self, pose_data: PoseData) -> Dict:
        """压缩姿态数据
        
        Args:
            pose_data: 姿态数据对象
            
        Returns:
            压缩后的数据字典
        """
        if not pose_data or not pose_data.keypoints:
            return {'error': 'No pose data'}
            
        # 1. 提取关键点坐标
        keypoints = []
        for i, kp in enumerate(pose_data.keypoints):
            keypoints.append([
                i,  # 关键点索引
                round(kp.get('x', 0), self.precision),
                round(kp.get('y', 0), self.precision),
                round(kp.get('z', 0) if 'z' in kp else 0, self.precision),
                round(kp.get('confidence', 1.0), self.precision)
            ])
            
        # 2. 转换为高效的数组格式
        data = {
            'keypoints': keypoints,
            'timestamp': pose_data.timestamp,
            'confidence': pose_data.confidence
        }
        
        return data
    
    def decompress_pose_data(self, compressed_data: Dict) -> Optional[PoseData]:
        """解压缩姿态数据
        
        Args:
            compressed_data: 压缩后的数据字典
            
        Returns:
            解压后的PoseData对象，失败返回None
        """
        try:
            if 'error' in compressed_data:
                return None
                
            # 重建关键点
            keypoints = []
            max_idx = -1
            
            # 先确定最大索引
            for item in compressed_data.get('keypoints', []):
                idx = item[0]
                max_idx = max(max_idx, idx)
                
            # 初始化关键点列表
            keypoints = [None] * (max_idx + 1)
            
            # 填充关键点
            for item in compressed_data.get('keypoints', []):
                idx, x, y, z, conf = item
                keypoints[idx] = {
                    'x': x,
                    'y': y,
                    'z': z,
                    'confidence': conf
                }
                
            # 补充没有的关键点
            for i, kp in enumerate(keypoints):
                if kp is None:
                    keypoints[i] = {'x': 0, 'y': 0, 'z': 0, 'confidence': 0}
            
            # 创建PoseData对象
            return PoseData(
                keypoints=keypoints,
                timestamp=compressed_data.get('timestamp', 0),
                confidence=compressed_data.get('confidence', 0)
            )
            
        except Exception as e:
            print(f"解压缩姿态数据失败: {e}")
            return None
    
    def serialize_for_transmission(self, data: Dict) -> str:
        """序列化数据用于传输
        
        Args:
            data: 要序列化的数据
            
        Returns:
            序列化并压缩后的字符串
        """
        # 转换为JSON字符串
        json_str = json.dumps(data)
        
        # 压缩并编码
        compressed = zlib.compress(json_str.encode('utf-8'))
        encoded = base64.b64encode(compressed).decode('ascii')
        
        return encoded
    
    def deserialize_from_transmission(self, encoded_str: str) -> Dict:
        """从传输数据反序列化
        
        Args:
            encoded_str: 编码后的字符串
            
        Returns:
            解码后的数据字典
        """
        try:
            # 解码并解压
            decoded = base64.b64decode(encoded_str)
            decompressed = zlib.decompress(decoded)
            
            # 转换回字典
            data = json.loads(decompressed.decode('utf-8'))
            
            return data
        except Exception as e:
            print(f"反序列化失败: {e}")
            return {'error': str(e)}
    
    def estimate_bandwidth(self, pose_data: PoseData, fps: int = 30) -> Dict:
        """估算传输带宽需求
        
        Args:
            pose_data: 姿态数据
            fps: 每秒帧数
            
        Returns:
            带宽估算信息
        """
        compressed = self.compress_pose_data(pose_data)
        serialized = self.serialize_for_transmission(compressed)
        
        bytes_per_frame = len(serialized)
        bytes_per_second = bytes_per_frame * fps
        
        return {
            'bytes_per_frame': bytes_per_frame,
            'bytes_per_second': bytes_per_second,
            'kbps': (bytes_per_second * 8) / 1000,  # 转为Kbps
            'compression_ratio': bytes_per_frame / (len(pose_data.keypoints) * 20)  # 假设原始数据每点20字节
        }

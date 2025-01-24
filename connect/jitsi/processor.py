from typing import Dict, List, Any, Optional
import numpy as np
import zlib
import json
import time
import logging
from dataclasses import dataclass, asdict

from lib.jitsi import JitsiError

logger = logging.getLogger(__name__)

@dataclass
class PoseData:
    """姿态数据结构"""
    keypoints: np.ndarray  # shape: (17, 3) - x, y, confidence
    bbox: List[float]      # [x, y, width, height]
    score: float          # 检测置信度
    timestamp: int        # 时间戳

class JitsiPoseProcessor:
    """Jitsi 姿态数据处理器"""
    
    def __init__(self, config: Dict):
        self.config = config.get('processor', {})
        self._compression_level = self.config.get('compression_level', 6)
        self._max_pose_size = self.config.get('max_pose_size', 1024 * 1024)
        self._batch_size = self.config.get('batch_size', 10)
        self._timeout = self.config.get('processing_timeout', 5.0)
        
        self._metrics = {
            'processed_count': 0,
            'compression_ratio': 0.0,
            'processing_time': 0.0,
            'errors': 0
        }

    def compress(self, pose_data: Dict) -> bytes:
        """压缩单个姿态数据"""
        try:
            if not self.validate_pose(pose_data):
                raise ValueError("Invalid pose data format")
                
            start_time = time.time()
            
            # 转换为内部数据结构
            pose = PoseData(
                keypoints=np.array(pose_data['keypoints']),
                bbox=pose_data['bbox'],
                score=pose_data['score'],
                timestamp=pose_data['timestamp']
            )
            
            # 序列化
            data = self._serialize_pose(pose)
            
            # 压缩
            compressed = self._compress_data(data)
            
            if len(compressed) > self._max_pose_size:
                raise ValueError("Compressed data exceeds size limit")
                
            # 更新指标
            self._update_metrics(
                original_size=len(data),
                compressed_size=len(compressed),
                process_time=time.time() - start_time
            )
            
            return compressed
            
        except Exception as e:
            self._metrics['errors'] += 1
            logger.error(f"Compression failed: {e}")
            raise JitsiError(f"Failed to compress pose data: {str(e)}")

    def decompress(self, data: bytes) -> Dict:
        """解压单个姿态数据"""
        try:
            # 解压缩
            decompressed = self._decompress_data(data)
            
            # 反序列化
            pose = self._deserialize_pose(decompressed)
            
            # 转换为字典格式
            return asdict(pose)
            
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise JitsiError(f"Failed to decompress data: {str(e)}")

    def compress_batch(self, poses: List[Dict]) -> bytes:
        """批量压缩姿态数据"""
        if len(poses) > self._batch_size:
            raise ValueError(f"Batch size exceeds limit: {self._batch_size}")
            
        compressed_poses = [self.compress(pose) for pose in poses]
        return self._compress_data(compressed_poses)

    def decompress_batch(self, data: bytes) -> List[Dict]:
        """批量解压姿态数据"""
        compressed_poses = self._decompress_data(data)
        return [self.decompress(pose) for pose in compressed_poses]

    def validate_pose(self, pose: Dict) -> bool:
        """验证姿态数据格式"""
        try:
            # 检查必要字段
            if not all(k in pose for k in ['keypoints', 'bbox', 'score', 'timestamp']):
                return False
                
            # 验证关键点数据
            keypoints = np.array(pose['keypoints'])
            if keypoints.shape != (17, 3):
                return False
                
            # 验证边界框
            if len(pose['bbox']) != 4 or any(x < 0 for x in pose['bbox']):
                return False
                
            # 验证置信度
            if not 0 <= pose['score'] <= 1:
                return False
                
            return True
            
        except Exception:
            return False

    def get_metrics(self) -> Dict:
        """获取性能指标"""
        return self._metrics.copy()

    def _serialize_pose(self, pose: PoseData) -> bytes:
        """序列化姿态数据"""
        data = {
            'keypoints': pose.keypoints.tolist(),
            'bbox': pose.bbox,
            'score': pose.score,
            'timestamp': pose.timestamp
        }
        return json.dumps(data).encode()

    def _deserialize_pose(self, data: bytes) -> PoseData:
        """反序列化姿态数据"""
        pose_dict = json.loads(data.decode())
        return PoseData(
            keypoints=np.array(pose_dict['keypoints']),
            bbox=pose_dict['bbox'],
            score=pose_dict['score'],
            timestamp=pose_dict['timestamp']
        )

    def _compress_data(self, data: bytes) -> bytes:
        """压缩数据"""
        return zlib.compress(data, level=self._compression_level)

    def _decompress_data(self, data: bytes) -> bytes:
        """解压数据"""
        return zlib.decompress(data)

    def _update_metrics(self, original_size: int, compressed_size: int, process_time: float):
        """更新性能指标"""
        self._metrics['processed_count'] += 1
        self._metrics['compression_ratio'] = compressed_size / original_size
        self._metrics['processing_time'] = process_time 
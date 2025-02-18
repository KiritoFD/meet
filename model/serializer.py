import zlib
import msgpack
import numpy as np
from pose.skeleton import Skeleton
from pose.binding import SkeletonBinding
import cv2

class ModelSerializer:
    """模型序列化器，支持压缩和分块传输"""
    
    def serialize(self, binding: SkeletonBinding) -> bytes:
        """序列化骨骼绑定数据"""
        data = {
            'bones': self._serialize_bones(binding.bones),
            'weights': self._serialize_weights(binding.weights),
            'mesh': self._serialize_mesh(binding.mesh_points),
            'texture': self._compress_texture(binding.texture)
        }
        packed = msgpack.packb(data, use_bin_type=True)
        return zlib.compress(packed, level=9)
    
    def _serialize_bones(self, bones):
        return [{
            'start_idx': b.start_idx,
            'end_idx': b.end_idx,
            'children': b.children,
            'influence_radius': b.influence_radius,
            'bind_matrix': b.bind_matrix.tobytes()
        } for b in bones]
    
    def _serialize_weights(self, weights):
        return {
            'dtype': str(weights.dtype),
            'shape': weights.shape,
            'data': weights.tobytes()
        }
    
    def _serialize_mesh(self, vertices):
        return {
            'vertices': vertices.tobytes(),
            'dtype': str(vertices.dtype),
            'shape': vertices.shape
        }
    
    def _compress_texture(self, texture):
        return zlib.compress(cv2.imencode('.webp', texture, [cv2.IMWRITE_WEBP_QUALITY, 90])[1]) 
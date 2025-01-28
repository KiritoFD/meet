import zlib
import msgpack
import numpy as np
import cv2
import torch
from pose.skeleton import Skeleton
from pose.binding import SkeletonBinding
from model.serializer import ModelSerializer

class ModelLoader:
    def __init__(self, gpu_device='cuda:0'):
        self.device = torch.device(gpu_device)
        self.serializer = ModelSerializer()
        
    def load_model(self, data: bytes) -> SkeletonBinding:
        """加载并初始化模型"""
        decompressed = zlib.decompress(data)
        model_data = msgpack.unpackb(decompressed, raw=False)
        
        binding = SkeletonBinding(
            bones=self._deserialize_bones(model_data['bones']),
            weights=self._deserialize_weights(model_data['weights']),
            mesh_points=self._deserialize_mesh(model_data['mesh']),
            texture=self._decompress_texture(model_data['texture'])
        )
        
        # 上传数据到GPU
        binding.weights = torch.tensor(binding.weights, device=self.device)
        binding.mesh_points = torch.tensor(binding.mesh_points, device=self.device)
        return binding
    
    def _deserialize_bones(self, bones_data):
        return [Bone(
            start_idx=b['start_idx'],
            end_idx=b['end_idx'],
            children=b['children'],
            influence_radius=b['influence_radius'],
            bind_matrix=np.frombuffer(b['bind_matrix'], dtype=np.float32).reshape(4,4)
        ) for b in bones_data]
    
    def _deserialize_weights(self, weights_data):
        return np.frombuffer(weights_data['data'], 
                            dtype=np.dtype(weights_data['dtype'])).reshape(weights_data['shape'])
    
    def _deserialize_mesh(self, mesh_data):
        return np.frombuffer(mesh_data['vertices'], 
                           dtype=np.dtype(mesh_data['dtype'])).reshape(mesh_data['shape'])
    
    def _decompress_texture(self, texture_data):
        return cv2.imdecode(zlib.decompress(texture_data), cv2.IMREAD_UNCHANGED) 
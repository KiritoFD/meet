import numpy as np
from scipy.spatial import cKDTree
import logging
from typing import List, Dict, Tuple, Optional
from .config import WeightingConfig, ComputeConfig

class OptimizedWeightCalculator:
    """优化的权重计算器"""
    
    def __init__(self, weight_config: WeightingConfig, compute_config: ComputeConfig):
        """初始化权重计算器
        
        Args:
            weight_config: 权重计算配置
            compute_config: 计算性能配置
        """
        self.weight_config = weight_config
        self.compute_config = compute_config
        self.logger = logging.getLogger(__name__)
        
        # 检测GPU可用性
        self.gpu_available = False
        if self.compute_config.use_gpu:
            try:
                import cupy as cp
                self._init_gpu()
                self.gpu_available = True
                self.logger.info("成功初始化GPU加速")
            except ImportError:
                self.logger.warning("未检测到CUDA环境，将使用CPU模式")
            except Exception as e:
                self.logger.warning(f"GPU初始化失败: {e}，将使用CPU模式")
    
    def _init_gpu(self):
        """初始化GPU相关资源"""
        pass  # TODO: 实现GPU初始化
        
    def _build_optimized_graph(self, vertices: np.ndarray) -> Dict:
        """构建优化的顶点图
        
        Args:
            vertices: 顶点坐标数组 (N, 3)
            
        Returns:
            包含图信息的字典
        """
        # 使用KD树加速最近邻搜索
        tree = cKDTree(vertices)
        
        # 为每个顶点找到最近的邻居
        distances, indices = tree.query(
            vertices, 
            k=min(10, len(vertices)),  # 每个顶点连接到最近的10个顶点
            distance_upper_bound=self.weight_config.falloff_radius
        )
        
        return {
            'tree': tree,
            'distances': distances,
            'indices': indices
        }
        
    def _process_single_bone(self,
                           joint_idx: int,
                           joint: 'Joint',
                           vertices: np.ndarray,
                           vertex_graph: Dict) -> np.ndarray:
        """处理单个骨骼的权重计算
        
        Args:
            joint_idx: 骨骼索引
            joint: 骨骼对象
            vertices: 顶点坐标数组
            vertex_graph: 顶点图信息
            
        Returns:
            该骨骼对所有顶点的权重值
        """
        # 计算到骨骼的距离
        distances = np.linalg.norm(
            vertices - joint.global_position,
            axis=1
        )
        
        # 应用距离衰减
        weights = np.exp(-distances * self.weight_config.distance_power)
        
        # 归一化
        weights /= np.sum(weights)
        
        return weights
        
    def compute_weights(self,
                       vertices: np.ndarray,
                       joints: List['Joint'],
                       cache_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """计算顶点权重
        
        Args:
            vertices: 顶点坐标数组 (N, 3)
            joints: 骨骼列表
            cache_path: 可选的缓存文件路径
            
        Returns:
            bone_ids: 每个顶点影响的骨骼ID (N, max_influences)
            weights: 对应的权重值 (N, max_influences)
        """
        vertex_count = len(vertices)
        joint_count = len(joints)
        
        # 构建优化图
        vertex_graph = self._build_optimized_graph(vertices)
        
        # 初始化结果数组
        bone_ids = np.zeros((vertex_count, self.weight_config.max_influences), dtype=np.int32)
        weights = np.zeros((vertex_count, self.weight_config.max_influences), dtype=np.float32)
        
        # 计算每个骨骼的权重
        all_weights = np.zeros((vertex_count, joint_count), dtype=np.float32)
        for i, joint in enumerate(joints):
            all_weights[:, i] = self._process_single_bone(i, joint, vertices, vertex_graph)
        
        # 选择最大的N个权重
        top_indices = np.argsort(all_weights, axis=1)[:, -self.weight_config.max_influences:]
        for i in range(vertex_count):
            bone_ids[i] = top_indices[i]
            weights[i] = all_weights[i, top_indices[i]]
            
        # 归一化权重
        weights /= np.sum(weights, axis=1, keepdims=True)
        
        return bone_ids, weights 
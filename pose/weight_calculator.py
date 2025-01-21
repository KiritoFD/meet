from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy.spatial import cKDTree
import json
import lz4.frame
from dataclasses import dataclass
import logging
import cupy as cp
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

@dataclass
class WeightingConfig:
    """权重计算配置"""
    max_influences: int = 4            # 每个顶点最大影响骨骼数
    falloff_radius: float = 0.1        # 影响衰减半径
    heat_iterations: int = 50          # 热扩散迭代次数
    smoothing_iterations: int = 2      # 平滑迭代次数
    distance_power: float = 2.0        # 距离权重指数
    volume_preservation: bool = True   # 是否保持体积

@dataclass
class ComputeConfig:
    """计算配置"""
    use_gpu: bool = True              # 默认尝试使用GPU
    num_threads: int = 8              # 默认线程数
    batch_size: int = 1024           # 默认批处理大小
    use_fast_math: bool = True       
    parallel_bones: bool = True      

class OptimizedWeightCalculator:
    """优化的权重计算器"""
    def __init__(self, weight_config: WeightingConfig, compute_config: ComputeConfig):
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
        """初始化GPU资源"""
        # CUDA kernel for heat diffusion
        self.heat_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void heat_diffusion(
            const float* vertex_positions,
            const int* graph_indices,
            const float* graph_weights,
            const int* graph_offsets,
            float* heat_values,
            const int num_vertices,
            const float diffusion_rate
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= num_vertices) return;
            
            float new_heat = 0.0f;
            float weight_sum = 0.0f;
            
            // Get neighbor range for current vertex
            int start = graph_offsets[idx];
            int end = graph_offsets[idx + 1];
            
            // Accumulate heat from neighbors
            for (int i = start; i < end; i++) {
                int neighbor = graph_indices[i];
                float weight = graph_weights[i];
                new_heat += heat_values[neighbor] * weight;
                weight_sum += weight;
            }
            
            // Normalize and apply diffusion
            if (weight_sum > 0.0f) {
                heat_values[idx] = new_heat / weight_sum;
            }
        }
        ''', 'heat_diffusion')
        
    def compute_weights(self,
                       vertices: np.ndarray,
                       joints: List['Joint'],
                       cache_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """计算权重（优化版本）"""
        if cache_path and self._load_from_cache(cache_path, vertices.shape[0]):
            return self.cached_bone_ids, self.cached_weights
            
        num_vertices = len(vertices)
        num_joints = len(joints)
        
        # 构建优化的顶点图
        vertex_graph = self._build_optimized_graph(vertices)
        
        # 初始化结果数组
        weights_full = np.zeros((num_vertices, num_joints), dtype=np.float32)
        
        if self.compute_config.parallel_bones:
            # 并行处理每个骨骼
            with ThreadPoolExecutor(max_workers=self.compute_config.num_threads) as executor:
                futures = []
                for joint_idx, joint in enumerate(joints):
                    future = executor.submit(
                        self._process_single_bone,
                        joint_idx,
                        joint,
                        vertices,
                        vertex_graph
                    )
                    futures.append(future)
                    
                # 收集结果
                for joint_idx, future in enumerate(futures):
                    weights_full[:, joint_idx] = future.result()
        else:
            # 串行处理
            for joint_idx, joint in enumerate(joints):
                weights_full[:, joint_idx] = self._process_single_bone(
                    joint_idx, joint, vertices, vertex_graph)
                
        # 优化的归一化和剪枝
        bone_ids, weights = self._optimized_normalize_and_prune(weights_full)
        
        if cache_path:
            self._save_to_cache(cache_path, bone_ids, weights)
            
        return bone_ids, weights
        
    def _build_optimized_graph(self, vertices: np.ndarray) -> Dict:
        """构建优化的顶点图"""
        # 使用KD树加速近邻搜索
        vertex_tree = cKDTree(vertices)
        
        # 预分配数组
        indices_list = []
        weights_list = []
        offsets = [0]
        
        # 批量处理顶点
        batch_size = self.compute_config.batch_size
        for i in range(0, len(vertices), batch_size):
            batch_vertices = vertices[i:i + batch_size]
            
            # 批量查询近邻
            distances, indices = vertex_tree.query(
                batch_vertices,
                k=10,
                distance_upper_bound=self.weight_config.falloff_radius
            )
            
            # 计算权重
            valid_mask = distances != np.inf
            batch_indices = indices[valid_mask]
            batch_distances = distances[valid_mask]
            
            batch_weights = np.exp(-batch_distances**2 / 
                                 (2 * self.weight_config.falloff_radius**2))
            
            indices_list.append(batch_indices)
            weights_list.append(batch_weights)
            
            # 更新偏移量
            for j in range(len(batch_vertices)):
                offsets.append(offsets[-1] + np.sum(valid_mask[j]))
                
        return {
            'indices': np.concatenate(indices_list),
            'weights': np.concatenate(weights_list).astype(np.float32),
            'offsets': np.array(offsets, dtype=np.int32)
        }
        
    def _process_single_bone(self,
                           joint_idx: int,
                           joint: 'Joint',
                           vertices: np.ndarray,
                           vertex_graph: Dict) -> np.ndarray:
        """处理单个骨骼的权重计算"""
        if self.gpu_available:
            return self._gpu_heat_diffusion(joint, vertices, vertex_graph)
        else:
            return self._cpu_heat_diffusion(joint, vertices, vertex_graph)
            
    def _gpu_heat_diffusion(self,
                           joint: 'Joint',
                           vertices: np.ndarray,
                           vertex_graph: Dict) -> np.ndarray:
        """GPU加速的热扩散"""
        num_vertices = len(vertices)
        
        # 转移数据到GPU
        d_vertices = cp.array(vertices)
        d_indices = cp.array(vertex_graph['indices'])
        d_weights = cp.array(vertex_graph['weights'])
        d_offsets = cp.array(vertex_graph['offsets'])
        
        # 初始化热量
        d_heat = cp.zeros(num_vertices, dtype=cp.float32)
        
        # 设置初始热源
        joint_pos = joint.global_position
        distances = cp.linalg.norm(d_vertices - joint_pos, axis=1)
        nearest_vertex = cp.argmin(distances)
        d_heat[nearest_vertex] = 1.0
        
        # 运行热扩散
        block_size = 256
        grid_size = (num_vertices + block_size - 1) // block_size
        
        for _ in range(self.weight_config.heat_iterations):
            self.heat_kernel(
                (grid_size,), (block_size,),
                (d_vertices, d_indices, d_weights, d_offsets,
                 d_heat, num_vertices, 0.5)
            )
            
        # 返回CPU
        return d_heat.get()
        
    def _cpu_heat_diffusion(self,
                           joint: 'Joint',
                           vertices: np.ndarray,
                           vertex_graph: Dict) -> np.ndarray:
        """CPU优化的热扩散"""
        num_vertices = len(vertices)
        heat = np.zeros(num_vertices, dtype=np.float32)
        
        # 设置初始热源
        joint_pos = joint.global_position
        distances = np.linalg.norm(vertices - joint_pos, axis=1)
        nearest_vertex = np.argmin(distances)
        heat[nearest_vertex] = 1.0
        
        # 使用NumPy优化的操作
        indices = vertex_graph['indices']
        weights = vertex_graph['weights']
        offsets = vertex_graph['offsets']
        
        for _ in range(self.weight_config.heat_iterations):
            new_heat = np.zeros_like(heat)
            
            # 并行处理顶点批次
            for i in range(0, num_vertices, self.compute_config.batch_size):
                end = min(i + self.compute_config.batch_size, num_vertices)
                
                for j in range(i, end):
                    start_idx = offsets[j]
                    end_idx = offsets[j + 1]
                    
                    if start_idx == end_idx:
                        continue
                        
                    neighbor_heat = heat[indices[start_idx:end_idx]]
                    neighbor_weights = weights[start_idx:end_idx]
                    
                    new_heat[j] = np.average(neighbor_heat, weights=neighbor_weights)
                    
            heat = new_heat
            
        return heat
        
    def _optimized_normalize_and_prune(self, 
                           weights_full: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """归一化并筛选最大影响"""
        num_vertices = weights_full.shape[0]
        
        # 找到每个顶点的最大影响骨骼
        top_indices = np.argsort(weights_full, axis=1)[:, -self.weight_config.max_influences:]
        top_weights = np.take_along_axis(
            weights_full, top_indices, axis=1)
            
        # 归一化权重
        weight_sums = np.sum(top_weights, axis=1, keepdims=True)
        top_weights /= weight_sums
        
        return top_indices, top_weights
        
    def _apply_volume_preservation(self,
                                 vertices: np.ndarray,
                                 joints: List['Joint'],
                                 bone_ids: np.ndarray,
                                 weights: np.ndarray) -> np.ndarray:
        """应用体积保持约束"""
        # 计算每个顶点的局部体积
        volumes = self._compute_local_volumes(vertices)
        
        # 调整权重以保持体积
        adjusted_weights = weights.copy()
        
        for i in range(len(vertices)):
            joint_indices = bone_ids[i]
            joint_dirs = np.array([joints[j].direction for j in joint_indices])
            
            # 计算体积保持因子
            volume_factors = np.abs(np.dot(joint_dirs, vertices[i]))
            volume_factors /= np.sum(volume_factors)
            
            # 调整权重
            adjusted_weights[i] *= volume_factors
            adjusted_weights[i] /= np.sum(adjusted_weights[i])
            
        return adjusted_weights
        
    def _save_to_cache(self, cache_path: str, bone_ids: np.ndarray, weights: np.ndarray):
        """压缩并保存权重到缓存"""
        cache_data = {
            'bone_ids': bone_ids.tolist(),
            'weights': weights.tolist(),
            'vertex_count': len(weights),
            'config': self.weight_config.__dict__
        }
        
        # 压缩JSON数据
        json_data = json.dumps(cache_data)
        compressed = lz4.frame.compress(json_data.encode())
        
        with open(cache_path, 'wb') as f:
            f.write(compressed)
            
    def _load_from_cache(self, cache_path: str, vertex_count: int) -> bool:
        """从缓存加载权重"""
        try:
            with open(cache_path, 'rb') as f:
                compressed = f.read()
                
            # 解压缩数据
            json_data = lz4.frame.decompress(compressed).decode()
            cache_data = json.loads(json_data)
            
            # 验证数据
            if cache_data['vertex_count'] != vertex_count:
                return False
                
            # 加载数据
            self.cached_bone_ids = np.array(cache_data['bone_ids'])
            self.cached_weights = np.array(cache_data['weights'])
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to load weight cache: {e}")
            return False 
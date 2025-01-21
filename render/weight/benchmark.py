import time
import numpy as np
import psutil
from typing import Dict, Any, List, Tuple
import logging
from .calculator import OptimizedWeightCalculator
from .config import WeightingConfig, ComputeConfig

def create_test_data(vertex_count: int, joint_count: int) -> Tuple[np.ndarray, List]:
    """创建测试用的顶点和骨骼数据"""
    vertices = np.random.rand(vertex_count, 3).astype(np.float32)
    joints = []
    for i in range(joint_count):
        joint = type('Joint', (), {
            'global_position': np.random.rand(3),
            'name': f'joint_{i}'
        })
        joints.append(joint)
    return vertices, joints

class WeightCalculatorBenchmark:
    """权重计算性能测试"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def run_benchmark(self, 
                     vertices: np.ndarray,
                     joints: list,
                     config: Dict[str, Any] = None) -> Dict[str, float]:
        """运行性能测试"""
        if config is None:
            config = self._get_default_config()
            
        # 创建计算器
        weight_config = WeightingConfig(**config.get('weight_config', {}))
        compute_config = ComputeConfig(**config.get('compute_config', {}))
        calculator = OptimizedWeightCalculator(weight_config, compute_config)
        
        results = {}
        
        # 记录基本信息
        results['vertex_count'] = len(vertices)
        results['joint_count'] = len(joints)
        results['gpu_available'] = calculator.gpu_available if hasattr(calculator, 'gpu_available') else False
        
        # 测试图构建性能
        start_time = time.perf_counter()
        vertex_graph = calculator._build_optimized_graph(vertices) if hasattr(calculator, '_build_optimized_graph') else None
        graph_time = time.perf_counter() - start_time
        results['graph_build_time'] = graph_time
        
        # 测试单骨骼处理性能
        start_time = time.perf_counter()
        if hasattr(calculator, '_process_single_bone'):
            calculator._process_single_bone(0, joints[0], vertices, vertex_graph)
        single_bone_time = time.perf_counter() - start_time
        results['single_bone_time'] = single_bone_time
        
        # 测试完整权重计算性能
        start_time = time.perf_counter()
        bone_ids, weights = calculator.compute_weights(vertices, joints)
        total_time = time.perf_counter() - start_time
        results['total_compute_time'] = total_time
        
        # 计算性能指标
        results['vertices_per_second'] = len(vertices) / total_time
        results['memory_usage'] = self._get_memory_usage()
        
        self._print_results(results)
        return results
        
    def _get_default_config(self) -> Dict[str, Dict]:
        """获取默认配置"""
        return {
            'weight_config': {
                'max_influences': 4,
                'falloff_radius': 0.1,
                'heat_iterations': 50
            },
            'compute_config': {
                'use_gpu': False,
                'num_threads': max(2, psutil.cpu_count(logical=False) - 1),
                'batch_size': 1024
            }
        }
        
    def _get_memory_usage(self) -> float:
        """获取当前内存使用"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
        
    def _print_results(self, results: Dict[str, float]):
        """打印测试结果"""
        print("\n=== 权重计算性能测试结果 ===")
        print(f"\n模型信息:")
        print(f"- 顶点数量: {results['vertex_count']}")
        print(f"- 骨骼数量: {results['joint_count']}")
        print(f"- GPU加速: {'可用' if results['gpu_available'] else '不可用'}")
        
        print("\n计算时间:")
        print(f"- 图构建时间: {results['graph_build_time']:.3f} 秒")
        print(f"- 单骨骼处理时间: {results['single_bone_time']:.3f} 秒")
        print(f"- 总计算时间: {results['total_compute_time']:.3f} 秒")
        
        print("\n性能指标:")
        print(f"- 每秒处理顶点数: {results['vertices_per_second']:.0f}")
        print(f"- 内存使用: {results['memory_usage']:.1f} MB") 
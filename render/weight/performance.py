import psutil
import numpy as np
import time
from typing import Dict, Tuple, Optional
from .benchmark import WeightCalculatorBenchmark, create_test_data

class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self):
        self.benchmark = WeightCalculatorBenchmark()
        self._system_info = self._get_system_info()
        
    def _get_system_info(self) -> Dict:
        """获取系统信息"""
        return {
            'cpu_count': psutil.cpu_count(logical=False),
            'total_memory': psutil.virtual_memory().total / (1024**3),  # GB
            'available_memory': psutil.virtual_memory().available / (1024**3),  # GB
            'cpu_freq': psutil.cpu_freq().max if psutil.cpu_freq() else None
        }
        
    def analyze_system(self, test_size: int = 5000) -> Dict:
        """分析系统性能并给出建议
        
        Args:
            test_size: 测试用的顶点数量
            
        Returns:
            性能分析结果和建议
        """
        print("\n=== 系统性能分析 ===")
        print(f"\n系统信息:")
        print(f"CPU核心数: {self._system_info['cpu_count']}")
        print(f"CPU频率: {self._system_info['cpu_freq']}MHz" if self._system_info['cpu_freq'] else "CPU频率: 未知")
        print(f"总内存: {self._system_info['total_memory']:.1f}GB")
        print(f"可用内存: {self._system_info['available_memory']:.1f}GB")
        
        # 运行基准测试
        vertices, joints = create_test_data(test_size, 10)
        results = self.benchmark.run_benchmark(vertices, joints)
        
        # 分析性能并给出建议
        recommendations = self._generate_recommendations(results)
        
        print("\n=== 性能建议 ===")
        for category, rec in recommendations.items():
            print(f"\n{category}:")
            for item in rec:
                print(f"- {item}")
                
        return recommendations
        
    def _generate_recommendations(self, results: Dict) -> Dict:
        """根据测试结果生成建议"""
        recs = {
            '推荐模型规模': [],
            '性能优化建议': [],
            '内存使用建议': []
        }
        
        # 计算推荐的模型规模
        vps = results['vertices_per_second']
        mem_per_vertex = results['memory_usage'] / results['vertex_count']
        available_mem = self._system_info['available_memory'] * 1024  # 转换为MB
        
        # 基于性能的推荐
        if vps > 100000:
            max_vertices_perf = int(vps * 0.5)  # 留出50%性能余量
        else:
            max_vertices_perf = int(vps * 0.7)  # 性能较低时留出30%余量
            
        # 基于内存的推荐
        max_vertices_mem = int(available_mem * 0.5 / mem_per_vertex)  # 最多使用50%可用内存
        
        # 取较小值作为最终推荐
        recommended_size = min(max_vertices_perf, max_vertices_mem)
        
        # 生成规模建议
        recs['推荐模型规模'].append(f"建议模型规模: {recommended_size:,} 顶点")
        recs['推荐模型规模'].append(f"最大安全规模: {int(recommended_size * 1.5):,} 顶点")
        
        # 性能优化建议
        if vps < 50000:
            recs['性能优化建议'].append("建议启用GPU加速")
            recs['性能优化建议'].append("考虑减少骨骼影响数量")
        elif vps < 100000:
            recs['性能优化建议'].append("可以适当增加批处理大小")
            
        # 内存使用建议
        if results['memory_usage'] / available_mem > 0.3:
            recs['内存使用建议'].append("建议启用内存优化模式")
            recs['内存使用建议'].append("考虑使用缓存机制")
            
        return recs
        
    def suggest_config(self, vertex_count: int) -> Tuple[Dict, str]:
        """根据目标顶点数量建议配置
        
        Args:
            vertex_count: 目标顶点数量
            
        Returns:
            推荐配置和风险等级
        """
        # 基于之前的性能测试结果估算资源使用
        test_results = self.analyze_system()
        
        # 评估风险等级
        risk_level = self._assess_risk(vertex_count, test_results)
        
        # 生成配置建议
        config = {
            'weight_config': {
                'max_influences': 4 if risk_level == 'low' else 3,
                'falloff_radius': 0.1,
                'heat_iterations': 50 if risk_level == 'low' else 30
            },
            'compute_config': {
                'use_gpu': risk_level == 'high',
                'num_threads': max(2, self._system_info['cpu_count'] - 1),
                'batch_size': 2048 if risk_level == 'low' else 1024
            }
        }
        
        return config, risk_level
        
    def _assess_risk(self, vertex_count: int, test_results: Dict) -> str:
        """评估处理特定规模模型的风险等级"""
        if vertex_count <= test_results.get('recommended_size', 10000):
            return 'low'
        elif vertex_count <= test_results.get('recommended_size', 10000) * 1.5:
            return 'medium'
        else:
            return 'high' 
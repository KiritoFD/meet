from dataclasses import dataclass

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
    use_gpu: bool = True              # 是否使用GPU
    num_threads: int = 8              # CPU线程数
    batch_size: int = 1024           # 批处理大小
    use_fast_math: bool = True       # 是否使用快速数学函数
    parallel_bones: bool = True      # 是否并行处理骨骼 
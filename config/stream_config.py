from dataclasses import dataclass

@dataclass
class StreamConfig:
    # 视频流配置
    quality: int = 85  # JPEG质量 (0-100)
    max_width: int = 640  # 最大宽度
    max_fps: int = 30  # 最大帧率
    
    # 处理选项
    enable_pose_overlay: bool = True  # 显示姿态关键点
    enable_smoothing: bool = True  # 启用平滑处理
    enable_gpu: bool = False  # 使用GPU加速
    
    # 网络选项
    chunk_size: int = 8192  # 块大小
    buffer_size: int = 5  # 缓冲区大小
    
    # 高级选项
    adaptive_quality: bool = True  # 根据网络自适应调整质量

# 默认流配置
DEFAULT_STREAM_CONFIG = StreamConfig()

# 高质量流配置
HIGH_QUALITY_STREAM_CONFIG = StreamConfig(
    quality=95,
    max_width=1280,
    max_fps=60
)

# 低带宽流配置
LOW_BANDWIDTH_STREAM_CONFIG = StreamConfig(
    quality=65,
    max_width=320,
    max_fps=15
)

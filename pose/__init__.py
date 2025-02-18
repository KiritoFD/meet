"""姿态处理模块"""
from .types import Landmark, PoseData
from .pose_binding import PoseBinding
from .binding import BindingConfig

# 2. 基础组件
from .smoother import FrameSmoother

# 3. 检测和处理组件
from .detector import PoseDetector
from .pose_deformer import PoseDeformer
from .deform_smoother import DeformSmoother

# 4. 管线和绘制
from .pipeline import PosePipeline
from .drawer import PoseDrawer

__all__ = [
    'Landmark',
    'PoseData',
    'PoseBinding',
    'BindingConfig',
    
    # 基础组件
    'FrameSmoother',
    
    # 检测和处理组件
    'PoseDetector',
    'PoseDeformer',
    'DeformSmoother',
    
    # 管线和绘制
    'PosePipeline',
    'PoseDrawer'
] 
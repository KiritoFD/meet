from typing import Dict, List, Optional, Callable
import numpy as np
from .controller import AnimationController

class AdvancedAnimationController:
    """高级动画控制器"""
    def __init__(self):
        self.base_controller = AnimationController()
        self.expression_weights = {}  # 表情权重
        self.motion_filters = {}      # 运动滤波器
        self.constraints = {}         # 参数约束
        
    def add_motion_filter(self, 
                         param_name: str, 
                         filter_func: Callable[[float], float]):
        """添加运动滤波器
        
        Args:
            param_name: 参数名称
            filter_func: 滤波函数
        """
        self.motion_filters[param_name] = filter_func
        
    def add_constraint(self, 
                      param_name: str, 
                      min_value: float = 0.0, 
                      max_value: float = 1.0):
        """添加参数约束
        
        Args:
            param_name: 参数名称
            min_value: 最小值
            max_value: 最大值
        """
        self.constraints[param_name] = (min_value, max_value)
        
    def update_expression(self, 
                         expression_params: Dict[str, float], 
                         blend_weight: float = 1.0):
        """更新表情参数
        
        Args:
            expression_params: 表情参数字典
            blend_weight: 混合权重
        """
        # 更新表情权重
        self.expression_weights = {
            name: value * blend_weight 
            for name, value in expression_params.items()
        }
        
        # 应用约束和滤波器
        filtered_params = {}
        for name, value in self.expression_weights.items():
            # 应用约束
            if name in self.constraints:
                min_val, max_val = self.constraints[name]
                value = np.clip(value, min_val, max_val)
                
            # 应用滤波器
            if name in self.motion_filters:
                value = self.motion_filters[name](value)
                
            filtered_params[name] = value
            
        # 设置动画目标
        for name, value in filtered_params.items():
            self.base_controller.set_target(name, value)
            
    def update(self) -> Dict[str, float]:
        """更新动画状态
        
        Returns:
            当前参数值字典
        """
        return self.base_controller.update() 
import numpy as np
from typing import List, Optional

def create_smooth_filter(window_size: int = 5) -> callable:
    """创建平滑滤波器
    
    Args:
        window_size: 窗口大小
        
    Returns:
        滤波函数
    """
    history: List[float] = []
    
    def smooth_filter(value: float) -> float:
        history.append(value)
        if len(history) > window_size:
            history.pop(0)
        return np.mean(history)
        
    return smooth_filter

def create_momentum_filter(momentum: float = 0.9) -> callable:
    """创建动量滤波器
    
    Args:
        momentum: 动量系数
        
    Returns:
        滤波函数
    """
    last_value: Optional[float] = None
    
    def momentum_filter(value: float) -> float:
        nonlocal last_value
        if last_value is None:
            last_value = value
        else:
            last_value = last_value * momentum + value * (1 - momentum)
        return last_value
        
    return momentum_filter 
import time
from dataclasses import dataclass
from typing import Dict, List
import psutil
import numpy as np

@dataclass
class PerformanceMetrics:
    fps: float
    latency: float
    success_rate: float
    cpu_usage: float
    memory_usage: float
    timestamp: float

class PerformanceMonitor:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history: List[PerformanceMetrics] = []
        self.start_time = time.time()
        self.frame_count = 0
        self.success_count = 0
        self._cleanup_interval = 1000  # Cleanup every 1000 updates
        self._update_count = 0
        
    def update(self, success: bool, latency: float):
        """更新性能指标"""
        self._update_count += 1
        self.frame_count += 1
        if success:
            self.success_count += 1
            
        metrics = PerformanceMetrics(
            fps=self._calculate_fps(),
            latency=latency,
            success_rate=self._calculate_success_rate(),
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.Process().memory_info().rss / 1024 / 1024,
            timestamp=time.time()
        )
        
        self.metrics_history.append(metrics)
        
        # Maintain window size
        while len(self.metrics_history) > self.window_size:
            self.metrics_history.pop(0)
            
        # Periodic cleanup
        if self._update_count >= self._cleanup_interval:
            self._cleanup()
            
    def _cleanup(self):
        """清理历史数据"""
        self._update_count = 0
        current_time = time.time()
        # Remove metrics older than 1 hour
        self.metrics_history = [
            m for m in self.metrics_history 
            if current_time - m.timestamp < 3600
        ]
        
    def get_stats(self) -> Dict[str, float]:
        """获取当前性能统计"""
        if not self.metrics_history:
            return {}
            
        latest = self.metrics_history[-1]
        return {
            'fps': latest.fps,
            'latency': latest.latency,
            'success_rate': latest.success_rate,
            'cpu_usage': latest.cpu_usage,
            'memory_usage': latest.memory_usage
        }
        
    def _calculate_fps(self) -> float:
        """计算当前帧率"""
        elapsed = time.time() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0
        
    def _calculate_success_rate(self) -> float:
        """计算成功率"""
        return self.success_count / self.frame_count if self.frame_count > 0 else 0 
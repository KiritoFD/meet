import time
import psutil
import os
from typing import Dict, Any
from collections import deque

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'fps': 0.0,
            'latency': 0.0,
            'success_rate': 1.0,
            'memory_usage': 0.0
        }
        self.frame_times = deque(maxlen=100)
        self.process = psutil.Process(os.getpid())

    def attach(self, sender):
        """附加到发送器以监控性能"""
        self.sender = sender
        sender.on_frame_sent = self._on_frame_sent

    def _on_frame_sent(self, success: bool, latency: float):
        """处理帧发送事件"""
        self.frame_times.append(time.time())
        if len(self.frame_times) >= 2:
            self.metrics['fps'] = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
        self.metrics['latency'] = latency
        self.metrics['memory_usage'] = self.process.memory_info().rss / 1024 / 1024  # MB

    def get_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        return self.metrics.copy() 
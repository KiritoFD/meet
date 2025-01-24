from dataclasses import dataclass
from typing import List, Dict, Optional
import time

@dataclass
class ConnectionMetrics:
    """连接指标"""
    latency: float           # 延迟(秒)
    packet_loss: float       # 丢包率(0-1)
    jitter: float           # 抖动(秒)
    bandwidth: float        # 带宽(bytes/s)
    state: str             # 连接状态
    timestamp: float = time.time()

@dataclass
class DataMetrics:
    """数据传输指标"""
    bytes_sent: int        # 发送字节数
    bytes_received: int    # 接收字节数
    messages_sent: int     # 发送消息数
    messages_received: int # 接收消息数
    compression_ratio: float  # 压缩比
    error_count: int = 0   # 错误计数
    timestamp: float = time.time()

@dataclass
class MeetingMetrics:
    """会议指标"""
    room_id: str          # 会议室ID
    participant_count: int # 参与者数量
    active_speakers: int  # 活跃发言者数量
    duration: float       # 会议持续时间(秒)
    peak_participants: int = 0  # 峰值参与者数量
    total_messages: int = 0    # 总消息数
    timestamp: float = time.time()

@dataclass
class SystemMetrics:
    """系统指标"""
    cpu_usage: float      # CPU使用率(0-1)
    memory_usage: float   # 内存使用率(0-1)
    thread_count: int     # 线程数
    disk_io: Optional[float] = None  # 磁盘IO(bytes/s)
    network_io: Optional[float] = None  # 网络IO(bytes/s)
    timestamp: float = time.time()

@dataclass
class PerformanceMetrics:
    """性能指标"""
    processing_time: float  # 处理时间(秒)
    queue_size: int        # 队列大小
    error_rate: float      # 错误率(0-1)
    timestamp: float = time.time()

class MetricsAggregator:
    """指标聚合器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._metrics: Dict[str, List] = {
            'connection': [],
            'data': [],
            'meeting': [],
            'system': [],
            'performance': []
        }

    def add_metrics(self, metric_type: str, metrics: any):
        """添加指标"""
        if metric_type not in self._metrics:
            raise ValueError(f"Unknown metric type: {metric_type}")
            
        self._metrics[metric_type].append(metrics)
        if len(self._metrics[metric_type]) > self.window_size:
            self._metrics[metric_type].pop(0)

    def get_metrics(self, metric_type: str) -> List:
        """获取指标"""
        return self._metrics.get(metric_type, [])

    def get_latest(self, metric_type: str) -> Optional[any]:
        """获取最新指标"""
        metrics = self._metrics.get(metric_type, [])
        return metrics[-1] if metrics else None

    def get_average(self, metric_type: str, field: str) -> Optional[float]:
        """获取指标平均值"""
        metrics = self._metrics.get(metric_type, [])
        if not metrics:
            return None
            
        values = [getattr(m, field) for m in metrics if hasattr(m, field)]
        return sum(values) / len(values) if values else None

    def get_summary(self) -> Dict:
        """获取指标汇总"""
        summary = {}
        for metric_type in self._metrics:
            metrics = self._metrics[metric_type]
            if not metrics:
                continue
                
            latest = metrics[-1]
            summary[metric_type] = {
                'count': len(metrics),
                'latest': latest,
                'timestamp': latest.timestamp
            }
        return summary

    def clear(self, metric_type: Optional[str] = None):
        """清除指标"""
        if metric_type:
            self._metrics[metric_type].clear()
        else:
            for metrics in self._metrics.values():
                metrics.clear() 
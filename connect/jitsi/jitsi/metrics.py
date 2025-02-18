from dataclasses import dataclass
from typing import List, Dict, Any
import time

@dataclass
class ConnectionMetrics:
    """连接指标"""
    latency: float
    packet_loss: float
    jitter: float
    bandwidth: float
    state: str
    timestamp: float = time.time()

@dataclass
class DataMetrics:
    """数据传输指标"""
    bytes_sent: int
    bytes_received: int
    messages_sent: int
    messages_received: int
    compression_ratio: float
    timestamp: float = time.time()

@dataclass
class MeetingMetrics:
    """会议指标"""
    room_id: str
    participant_count: int
    active_speakers: int
    duration: float
    timestamp: float = time.time()

@dataclass
class SystemMetrics:
    """系统指标"""
    cpu_usage: float
    memory_usage: float
    thread_count: int
    timestamp: float = time.time() 
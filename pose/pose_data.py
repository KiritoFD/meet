from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class PoseData:
    """姿态数据结构"""
    landmarks: List[Dict[str, float]]  # 关键点列表
    timestamp: float = 0.0  # 时间戳
    version: str = "1.0"   # 协议版本 
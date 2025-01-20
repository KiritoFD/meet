from dataclasses import dataclass
from typing import List, Optional
import time

@dataclass
class PoseData:
    keypoints: List[tuple]
    timestamp: float = time.time()
    confidence: float = 1.0 
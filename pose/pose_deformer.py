from typing import List, Optional
import numpy as np
from .pose_data import PoseData

class PoseDeformer:
    """姿态变形处理"""
    def __init__(self, smoothing_window: int = 5):
        self.smoothing_window = smoothing_window
        self._history = []

    def interpolate(self, pose1: PoseData, pose2: PoseData, t: float) -> PoseData:
        """姿态插值"""
        if pose1 is None or pose2 is None:
            raise ValueError("Invalid pose data")
            
        landmarks = []
        for lm1, lm2 in zip(pose1.landmarks, pose2.landmarks):
            landmark = {
                'x': lm1['x'] * (1-t) + lm2['x'] * t,
                'y': lm1['y'] * (1-t) + lm2['y'] * t,
                'z': lm1['z'] * (1-t) + lm2['z'] * t,
                'visibility': min(lm1['visibility'], lm2['visibility'])
            }
            landmarks.append(landmark)
            
        return PoseData(landmarks=landmarks)

    def smooth_sequence(self, poses: List[PoseData]) -> List[PoseData]:
        """平滑姿态序列"""
        if len(poses) < 3:
            raise ValueError("Sequence too short for smoothing")
            
        smoothed = []
        window = min(self.smoothing_window, len(poses))
        
        for i in range(len(poses)):
            start = max(0, i - window//2)
            end = min(len(poses), i + window//2 + 1)
            smoothed.append(self._average_poses(poses[start:end]))
            
        return smoothed

    def predict_next(self, history: List[PoseData]) -> Optional[PoseData]:
        """预测下一帧姿态"""
        if len(history) < 2:
            return None
            
        # 简单线性预测
        last = history[-1]
        prev = history[-2]
        
        landmarks = []
        for lm_last, lm_prev in zip(last.landmarks, prev.landmarks):
            dx = lm_last['x'] - lm_prev['x']
            dy = lm_last['y'] - lm_prev['y']
            dz = lm_last['z'] - lm_prev['z']
            
            landmark = {
                'x': lm_last['x'] + dx,
                'y': lm_last['y'] + dy,
                'z': lm_last['z'] + dz,
                'visibility': lm_last['visibility']
            }
            landmarks.append(landmark)
            
        return PoseData(landmarks=landmarks)

    def _average_poses(self, poses: List[PoseData]) -> PoseData:
        """计算多个姿态的平均"""
        n = len(poses)
        landmarks = []
        
        for i in range(len(poses[0].landmarks)):
            avg_lm = {
                'x': sum(p.landmarks[i]['x'] for p in poses) / n,
                'y': sum(p.landmarks[i]['y'] for p in poses) / n,
                'z': sum(p.landmarks[i]['z'] for p in poses) / n,
                'visibility': min(p.landmarks[i]['visibility'] for p in poses)
            }
            landmarks.append(avg_lm)
            
        return PoseData(landmarks=landmarks) 
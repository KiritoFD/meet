import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class PoseProcessor:
    def __init__(self, smooth_factor: float = 0.5):
        self.smooth_factor = smooth_factor
        self.last_pose = None
        
    def process_landmarks(self, results: Dict) -> Optional[Dict]:
        """处理检测结果，提取关键点"""
        try:
            pose_data = {}
            
            # 处理身体姿态
            if results['pose'] and results['pose'].pose_landmarks:
                pose_data['pose'] = [
                    {
                        'x': lm.x,
                        'y': lm.y,
                        'z': lm.z,
                        'visibility': lm.visibility
                    }
                    for lm in results['pose'].pose_landmarks.landmark
                ]
            
            # 处理面部网格
            if results['face_mesh'] and results['face_mesh'].multi_face_landmarks:
                pose_data['face'] = [
                    {
                        'x': lm.x,
                        'y': lm.y,
                        'z': lm.z
                    }
                    for lm in results['face_mesh'].multi_face_landmarks[0].landmark
                ]
            
            # 处理手部
            if results['hands'] and results['hands'].multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results['hands'].multi_hand_landmarks):
                    hand_data = [
                        {
                            'x': lm.x,
                            'y': lm.y,
                            'z': lm.z
                        }
                        for lm in hand_landmarks.landmark
                    ]
                    if idx == 0:
                        pose_data['left_hand'] = hand_data
                    else:
                        pose_data['right_hand'] = hand_data
            
            # 应用平滑
            if self.last_pose:
                pose_data = self._smooth_pose(pose_data)
            self.last_pose = pose_data
            
            return pose_data
            
        except Exception as e:
            logger.error(f"处理姿态数据失败: {str(e)}")
            return None
    
    def _smooth_pose(self, new_pose: Dict) -> Dict:
        """平滑姿态数据"""
        if not self.last_pose:
            return new_pose
            
        smoothed = {}
        for key in new_pose:
            if key in self.last_pose:
                smoothed[key] = self._smooth_landmarks(
                    self.last_pose[key],
                    new_pose[key]
                )
            else:
                smoothed[key] = new_pose[key]
                
        return smoothed
    
    def _smooth_landmarks(self, last: List, current: List) -> List:
        """平滑单个部位的关键点"""
        if len(last) != len(current):
            return current
            
        smoothed = []
        for l, c in zip(last, current):
            point = {}
            for k in c.keys():
                point[k] = l[k] * (1 - self.smooth_factor) + c[k] * self.smooth_factor
            smoothed.append(point)
            
        return smoothed 
import time
import random
import math

class AnimationController:
    def __init__(self):
        self.last_update_time = time.time()
        
        # 表情预设
        self.expressions = {
            'neutral': {
                'eye_open': 0.95,
                'eye_squint': 0.1,
                'mouth_smile': 0.2,
                'blush': 0.1,
                'brow_raise': 0.0,
                'brow_furrow': 0.0,
                'mouth_open': 0.0,
                'emotion': 'neutral'
            },
            'happy': {
                'eye_open': 0.7,
                'eye_squint': 0.4,
                'mouth_smile': 0.8,
                'blush': 0.6,
                'brow_raise': 0.3,
                'brow_furrow': 0.0,
                'mouth_open': 0.2,
                'emotion': 'happy'
            },
            'surprised': {
                'eye_open': 1.0,
                'eye_squint': 0.0,
                'mouth_smile': 0.0,
                'mouth_open': 0.7,
                'brow_raise': 0.9,
                'blush': 0.3,
                'brow_furrow': 0.0,
                'emotion': 'surprised'
            }
        }
        
        self.current_expression = 'neutral'
        self.target_expression = 'neutral'
        self.transition_progress = 1.0
        self.transition_duration = 0.5
        
    def update(self):
        current_time = time.time()
        dt = current_time - self.last_update_time
        
        if self.transition_progress < 1.0:
            self.transition_progress = min(1.0, 
                self.transition_progress + dt / self.transition_duration)
        
        values = self.expressions[self.current_expression].copy()
        
        if self.transition_progress < 1.0:
            target = self.expressions[self.target_expression]
            for key in values:
                if key != 'emotion':  # 跳过非数值属性
                    if key in target:
                        values[key] = self.lerp(values[key], target[key], 
                                              self.transition_progress)
            
            # 在过渡接近完成时更新emotion
            if self.transition_progress > 0.9:
                values['emotion'] = target['emotion']
        
        self.last_update_time = current_time
        return values
    
    def set_expression(self, expression, transition_time=0.5):
        if expression in self.expressions:
            self.current_expression = self.target_expression
            self.target_expression = expression
            self.transition_progress = 0.0
            self.transition_duration = transition_time
    
    @staticmethod
    def lerp(a, b, t):
        """仅对数值进行线性插值"""
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a + (b - a) * t
        return a  # 对于非数值类型，直接返回原值 
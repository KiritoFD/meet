import os
import cv2
import json
import logging
import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from .types import PoseData

logger = logging.getLogger(__name__)

@dataclass
class InitialFrame:
    image: np.ndarray
    keypoints: dict
    timestamp: float
    path: str

class InitialFrameManager:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.reference_dir = os.path.join(output_dir, 'reference')
        os.makedirs(self.reference_dir, exist_ok=True)
        self.initial_frame: Optional[InitialFrame] = None
        
    def save_initial_frame(self, 
                          frame: np.ndarray, 
                          pose_data: PoseData) -> Tuple[bool, str]:
        """保存初始参考帧"""
        try:
            # 检查输入
            if frame is None or pose_data is None:
                return False, "无效的输入数据"
                
            # 检查文件夹
            if not os.path.exists(self.reference_dir):
                os.makedirs(self.reference_dir)
                
            # 检查图像有效性
            if frame.size == 0 or len(frame.shape) != 3:
                return False, "无效的图像格式"
                
            # 保存图像
            frame_path = os.path.join(self.reference_dir, 'reference.jpg')
            success = cv2.imwrite(frame_path, frame)
            if not success:
                return False, "保存图像失败"
                
            # 转换关键点数据为可序列化格式
            keypoints_data = {
                'timestamp': float(pose_data.timestamp),  # 确保时间戳是浮点数
                'keypoints': []
            }
            
            for kp in pose_data.keypoints:
                keypoints_data['keypoints'].append({
                    'x': float(kp.x),
                    'y': float(kp.y),
                    'z': float(kp.z),
                    'visibility': float(kp.visibility)
                })
                
            # 保存关键点数据
            keypoints_path = os.path.join(self.reference_dir, 'keypoints.json')
            with open(keypoints_path, 'w', encoding='utf-8') as f:
                json.dump(keypoints_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"已保存参考帧: {frame_path}")
            logger.info(f"已保存关键点: {keypoints_path}")
            
            # 更新内存中的初始帧
            self.initial_frame = InitialFrame(
                image=frame.copy(),
                keypoints=keypoints_data,
                timestamp=float(pose_data.timestamp),
                path=frame_path
            )
            
            return True, frame_path
            
        except Exception as e:
            error_msg = f"保存初始帧失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg
    
    def load_initial_frame(self) -> Optional[InitialFrame]:
        """加载初始参考帧"""
        try:
            frame_path = os.path.join(self.reference_dir, 'reference.jpg')
            keypoints_path = os.path.join(self.reference_dir, 'keypoints.json')
            
            if not os.path.exists(frame_path) or not os.path.exists(keypoints_path):
                return None
            
            frame = cv2.imread(frame_path)
            with open(keypoints_path, 'r') as f:
                keypoints_data = json.load(f)
            
            self.initial_frame = InitialFrame(
                image=frame,
                keypoints=keypoints_data,
                timestamp=keypoints_data['timestamp'],
                path=frame_path
            )
            
            return self.initial_frame
            
        except Exception as e:
            logger.error(f"加载初始帧失败: {e}")
            return None
    
    def get_status(self) -> Dict:
        """获取初始帧状态"""
        frame_path = os.path.join(self.reference_dir, 'reference.jpg')
        keypoints_path = os.path.join(self.reference_dir, 'keypoints.json')
        
        has_frame = os.path.exists(frame_path)
        has_keypoints = os.path.exists(keypoints_path)
        
        timestamp = 0
        if has_keypoints and self.initial_frame:
            timestamp = self.initial_frame.timestamp
        
        return {
            'has_reference': has_frame,
            'has_keypoints': has_keypoints,
            'timestamp': timestamp,
            'path': frame_path if has_frame else None
        }

import cv2
import numpy as np
import logging
import json
import zlib
import base64
from pose.drawer import PoseDrawer
from utils.compression import decompress_data
from utils.image import decode_image

logger = logging.getLogger(__name__)

class ReceiveManager:
    def __init__(self):
        self.initial_image = None
        self.current_pose = None
        self.output_frame = None
        self.pose_drawer = PoseDrawer()
        
    def handle_initial_frame(self, data):
        """处理初始帧数据"""
        try:
            # 解码图像数据
            self.initial_image = decode_image(data['image'])
            logger.info('已接收初始帧')
        except Exception as e:
            logger.error(f"处理初始帧失败: {e}")
    
    def handle_pose_data(self, data):
        """处理姿态数据"""
        try:
            pose_data = self.decompress_pose_data(data['data'])
            if pose_data:
                self.current_pose = pose_data
                # 如果有初始帧，则绘制姿态
                if self.initial_image is not None:
                    self.output_frame = self.pose_drawer.draw_pose(
                        self.initial_image.copy(), 
                        self.current_pose
                    )
                logger.debug('已接收并处理姿态数据')
        except Exception as e:
            logger.error(f"处理姿态数据失败: {e}")
    
    @staticmethod
    def decompress_pose_data(compressed_data: bytes) -> dict:
        """解压缩姿态数据"""
        try:
            json_str = zlib.decompress(compressed_data).decode()
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"解压缩数据失败: {e}")
            return None
    
    def generate_frames(self):
        """生成视频帧"""
        while True:
            try:
                # 如果有输出帧，则发送
                if self.output_frame is not None:
                    ret, buffer = cv2.imencode('.jpg', self.output_frame)
                    if ret:
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
                # 添加适当的延迟
                cv2.waitKey(1)  # 使用OpenCV的等待函数
                
            except Exception as e:
                logger.error(f"生成帧错误: {e}")
                continue 
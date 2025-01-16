import logging
from typing import Optional
import cv2
import numpy as np
from .pose_sender import PoseSender
from .room_manager import RoomManager

class FrameProcessor:
    def __init__(self, pose_sender: PoseSender, room_manager: RoomManager):
        self.pose_sender = pose_sender
        self.room_manager = room_manager
        self.logger = logging.getLogger(__name__)

    async def process_frame(self, frame):
        """
        处理视频帧
        :param frame: 视频帧
        :return: 处理后的姿态数据
        """
        try:
            # 处理帧并获取姿态数据
            pose_data = self.detect_pose(frame)  # 假设这个方法存在
            if pose_data is None:
                return None
            
            # 获取当前房间ID
            current_room = self.room_manager.current_room
            
            # 使用 send_pose_data 方法发送数据
            await self.pose_sender.send_pose_data(
                pose_results=pose_data,
                room=current_room
            )
            
            return pose_data
        except Exception as e:
            self.logger.error(f"处理帧时出错: {str(e)}")
            return None

    def detect_pose(self, frame):
        """
        检测姿态
        这个方法应该根据你的具体实现来完成
        """
        # 你的姿态检测代码
        pass 
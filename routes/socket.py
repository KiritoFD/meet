import base64
import cv2
from flask import current_app
from utils.logger import logger

def init_socket_routes(socketio, camera_manager):
    @socketio.on('connect')
    def handle_connect():
        logger.info(f'客户端已连接')

    @socketio.on('request_frame')
    def handle_request_frame():
        """处理接收端请求帧"""
        with current_app.app_context():
            frame = camera_manager.current_frame
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_data = base64.b64encode(buffer.tobytes()).decode()
                    socketio.emit('frame_data', {'image': frame_data})

    @socketio.on('request_pose')
    def handle_request_pose():
        """处理接收端请求姿态数据"""
        with current_app.app_context():
            pose_data = camera_manager.current_pose
            if pose_data is not None:
                socketio.emit('pose_data', {'pose': pose_data}) 
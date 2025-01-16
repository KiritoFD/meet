from flask_socketio import SocketIO, emit, join_room, leave_room
from .room_manager import RoomManager
import logging
import base64
import numpy as np
import time

logger = logging.getLogger(__name__)

class SocketManager:
    def __init__(self, socketio: SocketIO, audio_processor):
        self.socketio = socketio
        self.audio_processor = audio_processor
        self.room_manager = RoomManager()
        self.setup_handlers()
        self.last_audio_log = 0  # 用于控制日志频率
    
    def setup_handlers(self):
        """设置Socket.IO事件处理器"""
        
        @self.socketio.on('connect')
        def handle_connect(auth):
            try:
                sid = self.socketio.request.sid
                logger.info(f"新用户连接: {sid}")
            except Exception as e:
                logger.error(f"处理连接时出错: {e}")
        
        @self.socketio.on('disconnect')
        def handle_disconnect(sid=None):
            try:
                sid = sid or self.socketio.request.sid
                logger.info(f"用户断开连接: {sid}")
                # 如果有正在进行的录音，停止它
                if self.audio_processor.is_recording:
                    self.audio_processor.stop_recording()
            except Exception as e:
                logger.error(f"处理断开连接时出错: {e}")
        
        @self.socketio.on('join_room')
        def handle_join_room(data):
            try:
                room_id = str(data.get('room_id'))
                if not room_id:
                    raise ValueError("房间ID不能为空")
                
                sid = self.socketio.sid
                
                # 加入Socket.IO房间
                join_room(room_id)
                
                # 加入房间管理器
                if self.room_manager.join_room(room_id, sid):
                    # 通知房间其他成员
                    emit('user_joined', {
                        'user': sid,
                        'room': room_id
                    }, room=room_id)
                    
                    # 通知当前用户
                    emit('room_joined', {
                        'room': room_id,
                        'members': list(self.room_manager.get_room_members(room_id))
                    })
                    
                    logger.info(f"用户 {sid} 成功加入房间 {room_id}")
                else:
                    emit('error', {'message': '加入房间失败'})
                    
            except Exception as e:
                logger.error(f"加入房间错误: {str(e)}")
                emit('error', {'message': str(e)})
        
        @self.socketio.on('leave_room')
        def handle_leave_room():
            sid = self.socketio.sid
            current_room = self.room_manager.get_user_room(sid)
            
            if current_room:
                # 离开Socket.IO房间
                leave_room(current_room)
                
                # 离开房间管理器
                self.room_manager.leave_current_room(sid)
                
                # 通知房间其他成员
                emit('user_left', {
                    'user': sid,
                    'room': current_room
                }, room=current_room)
                
                # 通知当前用户
                emit('room_left', {'room': current_room})
                
                logger.info(f"用户 {sid} 离开房间 {current_room}") 
        
        @self.socketio.on('video_frame')
        def handle_video_frame(data):
            try:
                # 获取音频数据
                audio_data = self.audio_processor.get_audio_data()
                
                # 每秒只打印一次音频数据信息
                current_time = time.time()
                if audio_data is not None:
                    if current_time - self.last_audio_log >= 1:
                        logger.info(f"音频数据大小: {len(audio_data)} bytes")
                        self.last_audio_log = current_time
                
                # 如果有音频数据，将其添加到视频帧数据中
                if audio_data:
                    data['audio'] = base64.b64encode(audio_data).decode('utf-8')
                    data['has_audio'] = True
                else:
                    data['has_audio'] = False
                
                # 广播给所有其他客户端
                emit('video_frame', data, broadcast=True, include_self=False)
            except Exception as e:
                logger.error(f"处理视频帧时出错: {e}") 
        
        @self.socketio.on('start_stream')
        def handle_start_stream():
            success = self.audio_processor.start_recording()
            emit('stream_status', {
                'status': 'started' if success else 'error',
                'message': '开始录音' if success else '启动失败'
            })
        
        @self.socketio.on('stop_stream')
        def handle_stop_stream():
            success, message = self.audio_processor.stop_recording()
            emit('stream_status', {
                'status': 'stopped' if success else 'error',
                'message': message
            })
        
        @self.socketio.on('request_audio_data')
        def handle_audio_request():
            """处理音频数据请求"""
            audio_data = self.audio_processor.get_audio_data()
            if audio_data:
                emit('audio_data', {
                    'audio': base64.b64encode(audio_data['data']).decode('utf-8'),
                    'level': audio_data['level'],
                    'sample_rate': audio_data['sample_rate']
                }) 
        
        @self.socketio.on_error()
        def error_handler(e):
            logger.error(f"Socket.IO 错误: {e}") 
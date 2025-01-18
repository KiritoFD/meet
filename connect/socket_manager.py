from flask_socketio import SocketIO, emit, join_room, leave_room
from .room_manager import RoomManager
import logging

logger = logging.getLogger(__name__)

class SocketManager:
    def __init__(self, socketio: SocketIO):
        self.socketio = socketio
        self.room_manager = RoomManager()
        self.setup_handlers()
    
    def setup_handlers(self):
        """设置Socket.IO事件处理器"""
        
        @self.socketio.on('connect')
        def handle_connect():
            logger.info(f"新用户连接: {self.socketio.sid}")
            emit('connected', {'sid': self.socketio.sid})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            sid = self.socketio.sid
            self.room_manager.leave_current_room(sid)
            logger.info(f"用户断开连接: {sid}")
        
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
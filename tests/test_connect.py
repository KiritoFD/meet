import pytest
import sys
from pathlib import Path
from flask import Flask

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from connect.room_manager import RoomManager
from connect.socket_manager import SocketManager
from connect.pose_sender import PoseSender, PoseData
import json
import zlib
from unittest.mock import Mock, patch
import numpy as np

# RoomManager测试
class TestRoomManager:
    def setup_method(self):
        self.room_manager = RoomManager()
    
    def test_join_room(self):
        # 测试加入房间
        assert self.room_manager.join_room("room1", "user1") == True
        assert "room1" in self.room_manager.rooms
        assert "user1" in self.room_manager.rooms["room1"]
        assert self.room_manager.user_rooms["user1"] == "room1"
    
    def test_leave_room(self):
        # 先加入房间
        self.room_manager.join_room("room1", "user1")
        # 测试离开房间
        self.room_manager.leave_current_room("user1")
        assert "user1" not in self.room_manager.user_rooms
        assert "room1" not in self.room_manager.rooms
    
    def test_multiple_users(self):
        # 测试多用户场景
        self.room_manager.join_room("room1", "user1")
        self.room_manager.join_room("room1", "user2")
        assert len(self.room_manager.rooms["room1"]) == 2
        
        self.room_manager.leave_current_room("user1")
        assert len(self.room_manager.rooms["room1"]) == 1
        assert "user2" in self.room_manager.rooms["room1"]

# SocketManager测试
class TestSocketManager:
    def setup_method(self):
        # 创建Flask应用
        self.app = Flask(__name__)
        self.mock_socketio = Mock()
        # 存储事件处理器
        self.handlers = {}
        
        # 模拟socketio.on装饰器
        def mock_on(event):
            def wrapper(f):
                self.handlers[event] = f
                return f
            return wrapper
        
        self.mock_socketio.on = mock_on
        
        # 模拟join_room和leave_room函数
        self.mock_join_room = Mock()
        self.mock_leave_room = Mock()
        
        self.socket_manager = SocketManager(self.mock_socketio)
    
    def test_setup_handlers(self):
        # 验证事件处理器是否正确设置
        handlers = ['connect', 'disconnect', 'join_room', 'leave_room']
        for handler in handlers:
            assert handler in self.handlers
    
    @patch('connect.socket_manager.emit')
    @patch('connect.socket_manager.join_room')
    def test_handle_join_room(self, mock_join_room, mock_emit):
        # 模拟join_room请求
        self.mock_socketio.sid = 'test_user'
        
        # 使用应用上下文
        with self.app.app_context():
            # 直接获取join_room处理器
            handler = self.handlers['join_room']
            handler({'room_id': 'test_room'})
        
        # 验证调用
        mock_join_room.assert_called_once_with('test_room')
        assert mock_emit.call_count >= 2
        assert any(call for call in mock_emit.call_args_list 
                  if call[0][0] == 'user_joined')
        assert any(call for call in mock_emit.call_args_list 
                  if call[0][0] == 'room_joined')
    
    @patch('connect.socket_manager.emit')
    @patch('connect.socket_manager.leave_room')
    def test_handle_leave_room(self, mock_leave_room, mock_emit):
        # 模拟用户在房间中
        self.mock_socketio.sid = 'test_user'
        self.socket_manager.room_manager.join_room('test_room', 'test_user')
        
        # 使用应用上下文
        with self.app.app_context():
            # 获取leave_room处理器
            handler = self.handlers['leave_room']
            
            # 执行离开房间
            handler()
        
        # 验证调用
        mock_leave_room.assert_called_once()
        assert mock_emit.call_count >= 2
        assert any(call for call in mock_emit.call_args_list 
                  if call[0][0] == 'user_left')
        assert any(call for call in mock_emit.call_args_list 
                  if call[0][0] == 'room_left')

# PoseSender测试
class TestPoseSender:
    def setup_method(self):
        self.mock_socketio = Mock()
        self.mock_room_manager = Mock()
        self.pose_sender = PoseSender(self.mock_socketio, self.mock_room_manager)
    
    def test_compress_data(self):
        test_data = {'test': 'data'}
        compressed = self.pose_sender._compress_data(test_data)
        decompressed = json.loads(zlib.decompress(compressed).decode())
        assert decompressed == test_data
    
    def test_has_significant_change(self):
        # 创建测试姿态数据
        old_pose = PoseData(pose_landmarks=[
            {'x': 0.5, 'y': 0.5, 'z': 0.0, 'visibility': 1.0}
        ])
        
        # 无变化的姿态
        new_pose = PoseData(pose_landmarks=[
            {'x': 0.5, 'y': 0.5, 'z': 0.0, 'visibility': 1.0}
        ])
        
        self.pose_sender.last_pose = old_pose
        assert not self.pose_sender._has_significant_change(new_pose)
        
        # 有显著变化的姿态
        new_pose = PoseData(pose_landmarks=[
            {'x': 0.6, 'y': 0.6, 'z': 0.1, 'visibility': 1.0}
        ])
        assert self.pose_sender._has_significant_change(new_pose)
    
    def test_send_pose_data(self):
        # 模拟MediaPipe结果
        mock_pose_results = Mock()
        mock_pose_results.pose_landmarks = Mock()
        mock_pose_results.pose_landmarks.landmark = [
            Mock(x=0.5, y=0.5, z=0.0, visibility=1.0)
        ]
        
        # 设置房间存在
        self.mock_room_manager.room_exists.return_value = True
        
        # 发送姿态数据
        self.pose_sender.send_pose_data(
            room="test_room",
            pose_results=mock_pose_results,
            face_results=None,
            hands_results=None,
            timestamp=123.456
        )
        
        # 验证数据发送
        assert self.mock_socketio.emit.called
        args = self.mock_socketio.emit.call_args[0]
        assert args[0] == 'pose_data'
        assert 'data' in args[1]

if __name__ == '__main__':
    pytest.main(['-v', 'test_connect.py']) 
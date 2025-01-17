import pytest
import time
import threading
from connect.socket_manager import SocketManager
from connect.room_manager import RoomManager
from connect.pose_sender import PoseSender

class TestStability:
    @pytest.mark.slow
    def test_long_running(self):
        """长时间运行测试"""
        socket = SocketManager()
        room_manager = RoomManager(socket)
        sender = PoseSender(socket)
        
        # 运行1小时
        end_time = time.time() + 3600
        errors = []
        
        def error_handler(e):
            errors.append(e)
        
        socket.on_error = error_handler
        
        while time.time() < end_time:
            sender.send_pose_data(
                "test_room",
                self._generate_test_pose()
            )
            time.sleep(1/30)  # 30fps
            
        assert len(errors) == 0
        assert socket.connected
        assert room_manager.current_room is not None

    @pytest.mark.slow
    def test_memory_usage(self):
        """内存使用测试"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # 运行系统10分钟
        socket = SocketManager()
        room_manager = RoomManager(socket)
        sender = PoseSender(socket)
        
        end_time = time.time() + 600
        while time.time() < end_time:
            sender.send_pose_data(
                "test_room",
                self._generate_test_pose()
            )
            time.sleep(1/30)
        
        # 验证内存增长不超过50MB
        final_memory = process.memory_info().rss
        assert (final_memory - initial_memory) < 50 * 1024 * 1024 
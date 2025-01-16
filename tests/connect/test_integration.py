import pytest
import time
from connect.core.socket_manager import SocketManager
from connect.core.room_manager import RoomManager
from connect.core.pose_sender import PoseSender

class TestIntegration:
    @pytest.fixture
    def setup_system(self):
        """初始化完整系统"""
        socket = SocketManager()
        room_manager = RoomManager(socket)
        sender = PoseSender(room_manager)
        
        socket.connect()
        yield (socket, room_manager, sender)
        socket.disconnect()

    def test_end_to_end(self, setup_system):
        """测试端到端流程"""
        socket, room_manager, sender = setup_system
        room_id = "test_room"
        
        # 1. 创建并加入房间
        assert room_manager.create_room(room_id)
        assert room_manager.join_room(room_id)
        
        # 2. 发送测试数据
        success_count = 0
        total_frames = 100
        
        for _ in range(total_frames):
            success = sender.send_frame(
                room_id=room_id,
                pose_results=self._generate_test_pose()
            )
            if success:
                success_count += 1
            time.sleep(0.033)  # ~30fps
            
        # 3. 验证性能指标
        stats = sender.get_stats()
        assert stats['fps'] >= 25
        assert stats['latency'] < 50
        assert success_count / total_frames > 0.99

    def test_error_recovery(self, setup_system):
        """测试错误恢复"""
        socket, room_manager, sender = setup_system
        
        # 1. 模拟断开连接
        socket.disconnect()
        time.sleep(0.1)
        
        # 2. 验证自动重连
        assert socket.connected
        
        # 3. 验证房间重新加入
        assert room_manager.current_room is not None
        
        # 4. 验证可以继续发送
        assert sender.send_frame(
            room_id=room_manager.current_room,
            pose_results=self._generate_test_pose()
        ) 
import pytest
import time
import numpy as np
from connect.socket_manager import SocketManager
from connect.room_manager import RoomManager
from connect.pose_sender import PoseSender
from connect.pose_protocol import PoseProtocol

class TestIntegration:
    @pytest.fixture
    def setup_system(self):
        """初始化完整系统"""
        socket = SocketManager()
        room_manager = RoomManager(socket)
        protocol = PoseProtocol()
        sender = PoseSender(socket, protocol)
        
        socket.connect()
        yield (socket, room_manager, sender, protocol)
        socket.disconnect()

    def test_end_to_end(self, setup_system):
        """测试完整流程"""
        socket, room_manager, sender, protocol = setup_system
        room_id = "test_room"
        
        # 1. 创建并加入房间
        assert room_manager.create_room(room_id)
        assert room_manager.join_room(room_id)
        
        # 2. 发送测试数据
        pose_data = self._generate_test_pose()
        success = sender.send_pose_data(room_id, pose_data)
        assert success
        
        # 3. 验证数据接收
        received_data = None
        @socket.on('pose_data')
        def handle_pose(data):
            nonlocal received_data
            received_data = protocol.decode(data)
        
        assert received_data is not None
        assert len(received_data.pose_landmarks) == len(pose_data.pose_landmarks)

    def test_stress(self, setup_system):
        """压力测试"""
        _, _, sender, _ = setup_system
        
        # 高频发送测试
        success_count = 0
        total_frames = 1000
        
        start_time = time.time()
        for _ in range(total_frames):
            if sender.send_pose_data("test_room", self._generate_test_pose()):
                success_count += 1
        
        duration = time.time() - start_time
        fps = total_frames / duration
        success_rate = success_count / total_frames
        
        assert fps >= 30  # 至少30fps
        assert success_rate >= 0.99  # 99%成功率

    def test_system_degradation(self):
        """测试系统降级机制"""
        socket, room_manager, sender, protocol = self.setup_system()
        
        # 模拟高负载
        for _ in range(1000):
            sender.send_pose_data("test_room", self._generate_test_pose())
        
        # 验证性能指标
        stats = sender.get_stats()
        assert stats['success_rate'] >= 0.95  # 允许5%的失败率
        
    def test_error_recovery_chain(self):
        """测试错误恢复链"""
        socket, room_manager, sender, protocol = self.setup_system()
        
        # 1. 断开连接
        socket.disconnect()
        
        # 2. 验证自动重连
        time.sleep(0.1)
        assert socket.connected
        
        # 3. 验证房间恢复
        assert room_manager.current_room is not None
        
        # 4. 验证数据发送恢复
        success = sender.send_pose_data(
            room_manager.current_room,
            self._generate_test_pose()
        )
        assert success

    @staticmethod
    def _generate_test_pose():
        """生成测试姿态数据"""
        return {
            'landmarks': [
                {
                    'x': np.random.random(),
                    'y': np.random.random(),
                    'z': np.random.random(),
                    'visibility': np.random.random()
                }
                for _ in range(33)
            ]
        } 
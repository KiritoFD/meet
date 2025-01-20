import pytest
import time
import asyncio
import numpy as np
from connect.socket_manager import SocketManager
from connect.room_manager import RoomManager
from connect.pose_sender import PoseSender
from connect.pose_protocol import PoseProtocol
from connect.unified_sender import UnifiedSender
from connect.performance_monitor import PerformanceMonitor
from connect.validator import DataValidator

class TestIntegration:
    @pytest.fixture
    async def setup_system(self):
        """初始化完整系统"""
        socket = SocketManager()
        room_manager = RoomManager(socket)
        protocol = PoseProtocol()
        sender = PoseSender(socket, protocol)
        monitor = PerformanceMonitor()
        validator = DataValidator()
        unified_sender = UnifiedSender(socket, monitor, validator)
        
        await socket.connect()
        yield {
            'socket': socket,
            'room': room_manager,
            'sender': sender,
            'protocol': protocol,
            'monitor': monitor,
            'validator': validator,
            'unified_sender': unified_sender
        }
        await socket.disconnect()

    @pytest.mark.asyncio
    async def test_end_to_end_flow(self, setup_system):
        """测试完整的端到端流程"""
        room_id = "test_room"
        user_id = "test_user"
        
        # 1. 创建并加入房间
        assert setup_system['room'].create_room(room_id)
        assert setup_system['room'].join_room(room_id, user_id)
        
        # 2. 发送数据
        pose_data = self._generate_test_pose()
        success = await setup_system['unified_sender'].send(
            data_type='pose',
            data=pose_data,
            room_id=room_id
        )
        assert success
        
        # 3. 检查性能指标
        stats = setup_system['monitor'].get_stats()
        assert stats['success_rate'] > 0.95
        assert stats['latency'] < 0.05
        
        # 4. 验证房间状态
        state = setup_system['room'].get_room_state(room_id)
        assert state['member_count'] == 1
        assert user_id in state['members']

    @pytest.mark.asyncio
    async def test_error_recovery(self, setup_system):
        """测试错误恢复流程"""
        # 1. 断开连接
        await setup_system['socket'].disconnect()
        
        # 2. 尝试发送数据
        pose_data = self._generate_test_pose()
        success = await setup_system['unified_sender'].send(
            data_type='pose',
            data=pose_data
        )
        
        # 3. 验证自动重连和恢复
        assert setup_system['socket'].connected
        assert success

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, setup_system):
        """测试并发操作"""
        room_id = "test_room"
        setup_system['room'].create_room(room_id)
        
        # 并发加入房间和发送数据
        async def user_session(user_id: str):
            assert setup_system['room'].join_room(room_id, user_id)
            pose_data = self._generate_test_pose()
            await setup_system['unified_sender'].send(
                data_type='pose',
                data=pose_data,
                room_id=room_id
            )
            
        # 创建多个用户会话
        tasks = [
            user_session(f"user_{i}")
            for i in range(10)
        ]
        await asyncio.gather(*tasks)
        
        # 验证结果
        state = setup_system['room'].get_room_state(room_id)
        assert state['member_count'] == 10

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
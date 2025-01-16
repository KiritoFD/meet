import pytest
import time
import numpy as np
from connect.core.pose_sender import PoseSender, SendConfig
from connect.core.room_manager import RoomManager
from connect.utils.errors import SendError, QueueFullError

class TestPoseSender:
    @pytest.fixture
    def setup_sender(self):
        """初始化测试环境"""
        room_manager = RoomManager(SocketManager())
        sender = PoseSender(room_manager)
        room_manager.socket.connect()
        yield sender
        room_manager.socket.disconnect()

    def test_send_performance(self, setup_sender):
        """测试发送性能"""
        sender = setup_sender
        room_id = "test_room"
        sender.room_manager.create_room(room_id)
        
        # 测试发送性能
        start_time = time.time()
        success_count = 0
        
        for _ in range(100):  # 测试100帧
            success = sender.send_frame(
                room_id=room_id,
                pose_results=self._generate_test_pose()
            )
            if success:
                success_count += 1
                
        duration = time.time() - start_time
        
        # 验证性能指标
        stats = sender.get_stats()
        assert stats['fps'] >= 25
        assert stats['latency'] < 50  # ms
        assert success_count / 100 > 0.99  # 99% 成功率

    def test_queue_management(self, setup_sender):
        """测试队列管理"""
        sender = setup_sender
        
        # 设置队列配置
        config = SendConfig(queue_size=5)
        sender.set_send_config(config)
        
        # 测试队列状态
        status = sender.get_queue_status()
        assert status['size'] == 0
        assert status['capacity'] == 5
        
        # 测试暂停/恢复
        sender.pause_sending()
        assert not sender._is_sending
        sender.resume_sending()
        assert sender._is_sending

    @staticmethod
    def _generate_test_pose():
        """生成测试姿态数据"""
        return {
            'landmarks': [
                {'x': np.random.random(), 
                 'y': np.random.random(),
                 'z': np.random.random()}
            ] * 33  # MediaPipe姿态点数量
        } 
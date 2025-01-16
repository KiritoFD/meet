import pytest
import asyncio
import logging
from connect.room_manager import RoomManager, RoomConfig
from connect.socket_manager import SocketManager
from connect.pose_sender import PoseSender
from connect.frame_processor import FrameProcessor

# 添加这行配置
pytestmark = pytest.mark.asyncio

class TestConnectIntegration:
    @pytest.fixture
    async def setup(self):
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        
        # 创建必要的组件
        socket_manager = SocketManager()
        room_config = RoomConfig(max_clients=10, timeout=30)
        room_manager = RoomManager(socket_manager, room_config)
        pose_sender = PoseSender(room_manager=room_manager, protocol=None)
        frame_processor = FrameProcessor(pose_sender, room_manager)
        
        yield {
            'socket_manager': socket_manager,
            'room_manager': room_manager,
            'pose_sender': pose_sender,
            'frame_processor': frame_processor
        }

    @pytest.mark.asyncio
    async def test_room_creation_and_joining(self, setup):
        room_manager = setup['room_manager']
        
        # 测试创建房间
        room_id = "test_room"
        assert room_manager.create_room(room_id) == True
        
        # 测试加入房间
        assert room_manager.join_room(room_id, "user1") == True
        
        # 验证房间状态
        room_status = room_manager.get_room_info(room_id)
        assert room_status is not None
        assert room_status.member_count == 1
        
        print("房间创建和加入测试通过")

    @pytest.mark.asyncio
    async def test_pose_sending(self, setup):
        room_manager = setup['room_manager']
        pose_sender = setup['pose_sender']
        
        # 创建测试房间
        room_id = "test_room_2"
        room_manager.create_room(room_id)
        room_manager.join_room(room_id, "user1")
        
        # 测试发送姿态数据
        test_pose_data = {
            "keypoints": [
                {"x": 0.5, "y": 0.5, "z": 0.0},
                {"x": 0.6, "y": 0.6, "z": 0.0}
            ]
        }
        
        success = await pose_sender.send_pose_data(
            pose_results=test_pose_data,
            room=room_id
        )
        
        assert success == True
        print("姿态数据发送测试通过")

    @pytest.mark.asyncio
    async def test_frame_processing(self, setup):
        frame_processor = setup['frame_processor']
        room_manager = setup['room_manager']
        
        # 创建测试房间
        room_id = "test_room_3"
        room_manager.create_room(room_id)
        room_manager.join_room(room_id, "user1")
        
        # 创建测试帧（模拟图像数据）
        test_frame = bytearray([0] * (640 * 480 * 3))  # 模拟 640x480 RGB 图像
        
        # 测试帧处理
        result = await frame_processor.process_frame(test_frame)
        print("帧处理测试完成")

    @pytest.mark.asyncio
    async def test_room_cleanup(self, setup):
        room_manager = setup['room_manager']
        
        # 创建测试房间
        room_id = "test_room_4"
        room_manager.create_room(room_id)
        room_manager.join_room(room_id, "user1")
        
        # 强制设置最后活跃时间为很久以前
        room_manager._rooms[room_id]['last_active'] = 0
        
        # 测试清理
        room_manager.clean_inactive_rooms()
        
        # 验证房间已被清理
        assert room_manager.get_room_info(room_id) is None
        print("房间清理测试通过")

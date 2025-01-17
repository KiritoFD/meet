import pytest
from unittest.mock import Mock
import time
from connect.room_manager import RoomManager, RoomConfig, RoomStatus, RoomMember
from connect.socket_manager import SocketManager
from connect.errors import RoomError, RoomNotFoundError, RoomFullError
from unittest.mock import patch

class TestRoomManager:
    @pytest.fixture
    def setup_room_manager(self):
        """初始化测试环境"""
        config = RoomConfig(max_clients=3, timeout=30)
        socket_manager = Mock(spec=SocketManager)
        room_manager = RoomManager(socket_manager, config)
        return room_manager

    def test_room_creation(self, setup_room_manager):
        """测试房间创建"""
        manager = setup_room_manager
        
        # 创建房间
        assert manager.create_room("test_room")
        assert manager.room_exists("test_room")
        
        # 重复创建
        with pytest.raises(RoomError):
            manager.create_room("test_room")

    def test_room_capacity(self, setup_room_manager):
        """测试房间容量限制"""
        manager = setup_room_manager
        room_id = "test_room"
        manager.create_room(room_id)

        # 添加最大数量的成员
        for i in range(3):
            assert manager.join_room(room_id, f"user{i}")

        # 超出容量
        with pytest.raises(RoomFullError):
            manager.join_room(room_id, "user4")

    def test_member_lifecycle(self, setup_room_manager):
        """测试成员生命周期"""
        manager = setup_room_manager
        room_id = "test_room"
        user_id = "test_user"

        manager.create_room(room_id)
        
        # 加入房间
        assert manager.join_room(room_id, user_id)
        assert user_id in manager.get_members(room_id)
        
        # 更新状态
        manager.update_member_status(room_id, user_id, "ready")
        member = manager.get_member(room_id, user_id)
        assert member.status == "ready"
        
        # 离开房间
        manager.leave_room(room_id, user_id)
        assert user_id not in manager.get_members(room_id)

    def test_room_cleanup(self, setup_room_manager):
        """测试房间清理"""
        manager = setup_room_manager
        room_id = "test_room"
        
        manager.create_room(room_id)
        manager.join_room(room_id, "user1")
        
        # 模拟超时
        time.sleep(0.1)
        manager._cleanup_rooms()
        
        assert not manager.room_exists(room_id)

    def test_room_events(self, setup_room_manager):
        """测试房间事件"""
        manager = setup_room_manager
        events = []

        @manager.on_room_event
        def handle_room_event(event_type, room_id, data):
            events.append((event_type, room_id, data))

        # 触发各种房间事件
        room_id = "test_room"
        manager.create_room(room_id)
        manager.join_room(room_id, "user1")
        manager.leave_room(room_id, "user1")

        assert len(events) == 3
        assert events[0][0] == "room_created"
        assert events[1][0] == "member_joined"
        assert events[2][0] == "member_left" 

    def test_room_state_sync(self):
        """测试房间状态同步"""
        manager = self.setup_room_manager()
        room_id = "test_room"
        
        # 模拟网络延迟后的状态同步
        with patch('time.sleep'):
            manager.create_room(room_id)
            manager.join_room(room_id, "user1")
            time.sleep(1)  # 模拟延迟
            
            # 验证状态一致性
            room_state = manager.get_room_state(room_id)
            assert room_state.is_consistent()

    def test_cleanup_mechanism(self):
        """测试完整清理机制"""
        manager = self.setup_room_manager()
        
        # 创建多个测试房间
        rooms = ["room1", "room2", "room3"]
        for room in rooms:
            manager.create_room(room)
            manager.join_room(room, f"user_{room}")
        
        # 模拟超时
        time.sleep(31)  # 配置的超时时间是30秒
        manager._cleanup_rooms()
        
        # 验证清理结果
        assert len(manager.rooms) == 0
        assert len(manager.user_rooms) == 0 
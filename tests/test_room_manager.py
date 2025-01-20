import pytest
import time
from unittest.mock import Mock
from connect.room_manager import RoomManager, RoomConfig, RoomStatus, RoomMember

class TestRoomManager:
    @pytest.fixture
    def socket_mock(self):
        socket = Mock()
        socket.sid = "test_user_id"
        socket.broadcast = Mock(return_value=True)
        return socket

    @pytest.fixture
    def room_manager(self, socket_mock):
        config = RoomConfig(
            max_clients=3,
            timeout=5,
            auto_cleanup=False
        )
        return RoomManager(socket_mock, config)

    def test_create_room(self, room_manager):
        # 测试创建房间
        assert room_manager.create_room("room1") == True
        # 测试重复创建
        assert room_manager.create_room("room1") == False

    def test_join_room(self, room_manager):
        # 创建房间
        room_manager.create_room("room1")
        
        # 测试加入房间
        assert room_manager.join_room("room1", "user1") == True
        assert room_manager.join_room("room1", "user2") == True
        
        # 验证成员数量
        members = room_manager.get_members("room1")
        assert len(members) == 2
        
        # 验证第一个加入的用户是主持人
        host = next(m for m in members if m.id == "user1")
        assert host.role == "host"

    def test_leave_room(self, room_manager):
        # 准备测试环境
        room_manager.create_room("room1")
        room_manager.join_room("room1", "user1")
        room_manager.join_room("room1", "user2")
        
        # 测试离开房间
        room_manager.leave_room("room1", "user1")
        members = room_manager.get_members("room1")
        assert len(members) == 1
        assert members[0].id == "user2"

    def test_room_max_clients(self, room_manager):
        # 测试房间人数上限
        room_manager.create_room("room1")
        assert room_manager.join_room("room1", "user1") == True
        assert room_manager.join_room("room1", "user2") == True
        assert room_manager.join_room("room1", "user3") == True
        # 第4个人应该加入失败
        assert room_manager.join_room("room1", "user4") == False

    def test_broadcast(self, room_manager, socket_mock):
        # 测试广播功能
        room_manager.create_room("room1")
        room_manager.join_room("room1", "user1")
        
        assert room_manager.broadcast("test_event", {"message": "hello"}, "room1") == True
        socket_mock.broadcast.assert_called_once_with("test_event", {"message": "hello"}, room="room1")

    def test_room_status(self, room_manager):
        # 测试房间状态
        room_manager.create_room("room1")
        room_manager.join_room("room1", "user1")
        
        status = room_manager.get_room_info("room1")
        assert isinstance(status, RoomStatus)
        assert status.room_id == "room1"
        assert status.member_count == 1
        assert status.is_locked == False

    def test_clean_inactive_rooms(self, room_manager):
        print("\nStarting clean inactive rooms test")  # 添加调试信息
        
        # 创建房间
        room_manager.create_room("room1")
        print("Room created")
        
        # 加入用户
        room_manager.join_room("room1", "user1")
        print("User joined")
        
        # 修改最后活跃时间
        current_time = time.time()
        room_manager._rooms["room1"]["last_active"] = current_time - 10
        print(f"Set last_active to {current_time - 10}")
        
        # 清理不活跃房间
        print("Starting cleanup")
        room_manager.clean_inactive_rooms()
        print("Cleanup completed")
        
        # 验证房间已被清理
        assert room_manager.get_room_info("room1") is None
        print("Verified room is cleaned")
        assert len(room_manager.get_members("room1")) == 0
        print("Verified members are cleaned")

    def test_member_role_management(self, room_manager):
        # 测试成员角色管理
        room_manager.create_room("room1")
        room_manager.join_room("room1", "user1")
        room_manager.join_room("room1", "user2")
        
        # 测试设置角色
        room_manager.set_member_role("room1", "user2", "host")
        members = room_manager.get_members("room1")
        user2 = next(m for m in members if m.id == "user2")
        assert user2.role == "host"

        # 测试无效角色
        with pytest.raises(ValueError):
            room_manager.set_member_role("room1", "user2", "invalid_role")

    def test_kick_member(self, room_manager):
        # 测试踢出成员
        room_manager.create_room("room1")
        room_manager.join_room("room1", "user1")
        room_manager.join_room("room1", "user2")
        
        room_manager.kick_member("room1", "user2")
        members = room_manager.get_members("room1")
        assert len(members) == 1
        assert members[0].id == "user1" 
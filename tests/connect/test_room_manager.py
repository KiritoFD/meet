import pytest
import time
from connect.core.room_manager import RoomManager, RoomConfig, RoomStatus, RoomMember
from connect.core.socket_manager import SocketManager
from connect.utils.errors import RoomError, RoomNotFoundError

class TestRoomManager:
    @pytest.fixture
    def setup_room(self):
        """初始化测试环境"""
        socket = SocketManager()
        config = RoomConfig(
            max_clients=5,
            timeout=10,
            auto_cleanup=True
        )
        manager = RoomManager(socket, config)
        socket.connect()
        yield manager
        socket.disconnect()

    def test_room_lifecycle(self, setup_room):
        """测试房间生命周期"""
        manager = setup_room
        room_id = "test_room"
        
        # 创建房间
        assert manager.create_room(room_id)
        
        # 加入房间
        assert manager.join_room(room_id)
        assert manager.current_room == room_id
        
        # 获取房间信息
        info = manager.get_room_info(room_id)
        assert isinstance(info, RoomStatus)
        assert info.room_id == room_id
        assert info.member_count == 1
        
        # 离开房间
        manager.leave_room(room_id)
        assert manager.current_room is None

    def test_member_management(self, setup_room):
        """测试成员管理"""
        manager = setup_room
        room_id = "test_room"
        
        manager.create_room(room_id)
        manager.join_room(room_id)
        
        # 获取成员列表
        members = manager.get_members(room_id)
        assert len(members) == 1
        assert isinstance(members[0], RoomMember)
        
        # 更新成员状态
        manager.update_member_status(room_id, members[0].id)
        
        # 设置角色
        manager.set_member_role(room_id, members[0].id, "host")
        updated_members = manager.get_members(room_id)
        assert updated_members[0].role == "host" 
import pytest
import time
from connect.room_manager import RoomManager
from connect.errors import RoomError

class TestRoomManager:
    @pytest.fixture
    def setup_room(self, mock_socket):
        """初始化房间管理器"""
        manager = RoomManager(mock_socket)
        yield manager
        manager.cleanup()

    def test_room_lifecycle(self, setup_room):
        """测试房间生命周期"""
        room_id = "test_room"
        
        # 创建房间
        assert setup_room.create_room(room_id)
        assert setup_room.room_exists(room_id)
        
        # 加入成员
        user_id = "test_user"
        assert setup_room.join_room(room_id, user_id)
        assert user_id in setup_room.get_members(room_id)
        
        # 离开房间
        setup_room.leave_room(room_id, user_id)
        assert user_id not in setup_room.get_members(room_id)
        
        # 关闭房间
        setup_room.close_room(room_id)
        assert not setup_room.room_exists(room_id)

    def test_room_events(self, setup_room):
        """测试房间事件系统"""
        events = []
        
        @setup_room.on('member_join')
        def on_join(room_id, user_id):
            events.append(('join', room_id, user_id))
            
        @setup_room.on('member_leave')
        def on_leave(room_id, user_id):
            events.append(('leave', room_id, user_id))
            
        # 触发事件
        room_id = "test_room"
        user_id = "test_user"
        
        setup_room.create_room(room_id)
        setup_room.join_room(room_id, user_id)
        setup_room.leave_room(room_id, user_id)
        
        assert len(events) == 2
        assert events[0][0] == 'join'
        assert events[1][0] == 'leave'

    def test_room_state_sync(self, setup_room):
        """测试房间状态同步"""
        room_id = "test_room"
        setup_room.create_room(room_id)
        
        # 添加测试成员
        for i in range(3):
            setup_room.join_room(room_id, f"user_{i}")
            
        # 获取状态
        state = setup_room.get_room_state(room_id)
        
        assert state['member_count'] == 3
        assert len(state['members']) == 3
        assert state['is_locked'] is False

    def test_room_cleanup(self, setup_room):
        """测试房间清理机制"""
        # 创建测试房间
        room_id = "test_room"
        setup_room.create_room(room_id)
        setup_room.join_room(room_id, "test_user")
        
        # 模拟超时
        setup_room._rooms[room_id]['last_active'] = time.time() - 3600
        
        # 执行清理
        setup_room.clean_inactive_rooms()
        
        assert not setup_room.room_exists(room_id) 
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from connect.jitsi.meeting import JitsiMeeting
from connect.jitsi.meeting_manager import JitsiMeetingManager

class TestJitsiMeeting:
    @pytest.fixture
    def meeting(self):
        return JitsiMeeting("test_room", "host_id")

    def test_add_participant(self, meeting):
        meeting.add_participant("user1")
        assert "user1" in meeting.participants

    def test_remove_participant(self, meeting):
        meeting.add_participant("user1")
        meeting.remove_participant("user1")
        assert "user1" not in meeting.participants

    def test_is_host(self, meeting):
        assert meeting.is_host("host_id")
        assert not meeting.is_host("user1")

@pytest.fixture
def mock_manager():
    config = {
        'jitsi_host': 'localhost',
        'max_participants': 10,
        'room_timeout': 300
    }
    return JitsiMeetingManager(config)

@pytest.mark.asyncio
async def test_create_meeting(mock_manager):
    """测试创建会议室"""
    mock_manager.client.join_room = AsyncMock()
    room_id = await mock_manager.create_meeting("host_123")
    assert room_id is not None
    assert room_id in mock_manager.active_rooms

@pytest.mark.asyncio
async def test_join_meeting_success(mock_manager):
    """测试成功加入会议室"""
    room_id = await mock_manager.create_meeting("host_123")
    result = await mock_manager.join_meeting(room_id, "user_456")
    assert result is True
    assert "user_456" in mock_manager.active_rooms[room_id].participants

@pytest.mark.asyncio
async def test_room_cleanup():
    """测试会议室自动清理"""
    config = {'room_timeout': 0.1}  # 设置超时时间为0.1秒
    manager = JitsiMeetingManager(config)
    room_id = await manager.create_meeting("host_123")
    
    await asyncio.sleep(0.2)  # 等待超时
    await manager._cleanup_idle_rooms()
    assert room_id not in manager.active_rooms 
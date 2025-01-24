import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import json

from connect.jitsi.meeting_manager import JitsiMeetingManager
from connect.jitsi.meeting import JitsiMeeting
from lib.jitsi import JitsiError

@pytest.fixture
def config():
    return {
        'jitsi': {
            'host': 'test.jitsi.com',
            'port': 443,
            'ice_servers': [
                {'urls': ['stun:stun.l.google.com:19302']}
            ]
        },
        'meeting': {
            'max_participants': 16,
            'idle_timeout': 300,
            'cleanup_interval': 60
        }
    }

@pytest.fixture
def mock_meeting():
    mock = AsyncMock()
    mock.room_id = "test_room"
    mock.host_id = "test_host"
    mock.get_participants = Mock(return_value=["test_host"])
    mock.is_active = Mock(return_value=True)
    mock.join = AsyncMock()
    mock.leave = AsyncMock()
    mock.close = AsyncMock()
    return mock

@pytest.fixture
async def manager(config):
    manager = JitsiMeetingManager(config)
    yield manager
    await manager.close()

class TestJitsiMeetingManager:
    @pytest.mark.asyncio
    async def test_create_meeting(self, manager, mock_meeting):
        """测试创建会议"""
        with patch('connect.jitsi.meeting_manager.JitsiMeeting', return_value=mock_meeting):
            room_id = await manager.create_meeting("test_host")
            
            assert room_id == mock_meeting.room_id
            assert room_id in manager.meetings
            assert manager.get_user_meeting("test_host") == room_id

    @pytest.mark.asyncio
    async def test_join_meeting(self, manager, mock_meeting):
        """测试加入会议"""
        with patch('connect.jitsi.meeting_manager.JitsiMeeting', return_value=mock_meeting):
            room_id = await manager.create_meeting("test_host")
            success = await manager.join_meeting(room_id, "test_user")
            
            assert success
            mock_meeting.join.assert_called_once_with("test_user")
            assert manager.get_user_meeting("test_user") == room_id

    @pytest.mark.asyncio
    async def test_leave_meeting(self, manager, mock_meeting):
        """测试离开会议"""
        with patch('connect.jitsi.meeting_manager.JitsiMeeting', return_value=mock_meeting):
            room_id = await manager.create_meeting("test_host")
            await manager.join_meeting(room_id, "test_user")
            
            await manager.leave_meeting("test_user")
            mock_meeting.leave.assert_called_once_with("test_user")
            assert manager.get_user_meeting("test_user") is None

    @pytest.mark.asyncio
    async def test_close_meeting(self, manager, mock_meeting):
        """测试关闭会议"""
        with patch('connect.jitsi.meeting_manager.JitsiMeeting', return_value=mock_meeting):
            room_id = await manager.create_meeting("test_host")
            await manager.close_meeting(room_id)
            
            mock_meeting.close.assert_called_once()
            assert room_id not in manager.meetings

    @pytest.mark.asyncio
    async def test_participant_limit(self, manager, mock_meeting):
        """测试参与者数量限制"""
        with patch('connect.jitsi.meeting_manager.JitsiMeeting', return_value=mock_meeting):
            room_id = await manager.create_meeting("test_host")
            
            # 模拟达到最大参与者数量
            mock_meeting.get_participants.return_value = ["test_host"] + [
                f"user_{i}" for i in range(manager.config['meeting']['max_participants'] - 1)
            ]
            
            # 尝试加入新用户应该失败
            success = await manager.join_meeting(room_id, "extra_user")
            assert not success

    @pytest.mark.asyncio
    async def test_idle_timeout(self, manager, mock_meeting):
        """测试空闲超时"""
        with patch('connect.jitsi.meeting_manager.JitsiMeeting', return_value=mock_meeting):
            room_id = await manager.create_meeting("test_host")
            
            # 模拟会议变为非活动状态
            mock_meeting.is_active.return_value = False
            
            # 等待清理周期
            await asyncio.sleep(0.1)
            await manager._cleanup_meetings()
            
            assert room_id not in manager.meetings

    @pytest.mark.asyncio
    async def test_host_privileges(self, manager, mock_meeting):
        """测试主持人权限"""
        with patch('connect.jitsi.meeting_manager.JitsiMeeting', return_value=mock_meeting):
            room_id = await manager.create_meeting("test_host")
            
            # 非主持人不能关闭会议
            with pytest.raises(ValueError):
                await manager.close_meeting(room_id, "test_user")
                
            # 主持人可以关闭会议
            await manager.close_meeting(room_id, "test_host")
            assert room_id not in manager.meetings

    @pytest.mark.asyncio
    async def test_meeting_stats(self, manager, mock_meeting):
        """测试会议统计"""
        with patch('connect.jitsi.meeting_manager.JitsiMeeting', return_value=mock_meeting):
            room_id = await manager.create_meeting("test_host")
            await manager.join_meeting(room_id, "test_user")
            
            stats = manager.get_stats()
            assert stats['total_meetings'] == 1
            assert stats['active_meetings'] == 1
            assert stats['total_participants'] == 2

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, manager, mock_meeting):
        """测试并发操作"""
        with patch('connect.jitsi.meeting_manager.JitsiMeeting', return_value=mock_meeting):
            room_id = await manager.create_meeting("test_host")
            
            # 并发加入多个用户
            join_tasks = [
                manager.join_meeting(room_id, f"user_{i}")
                for i in range(5)
            ]
            results = await asyncio.gather(*join_tasks)
            
            successful_joins = sum(1 for r in results if r)
            assert successful_joins == 5

    @pytest.mark.asyncio
    async def test_error_handling(self, manager, mock_meeting):
        """测试错误处理"""
        with patch('connect.jitsi.meeting_manager.JitsiMeeting', return_value=mock_meeting):
            room_id = await manager.create_meeting("test_host")
            
            # 模拟加入失败
            mock_meeting.join.side_effect = JitsiError("Join failed")
            
            with pytest.raises(JitsiError):
                await manager.join_meeting(room_id, "test_user")

    @pytest.mark.asyncio
    async def test_cleanup_on_close(self, manager, mock_meeting):
        """测试关闭时清理"""
        with patch('connect.jitsi.meeting_manager.JitsiMeeting', return_value=mock_meeting):
            room_id = await manager.create_meeting("test_host")
            await manager.close()
            
            mock_meeting.close.assert_called_once()
            assert len(manager.meetings) == 0
            assert len(manager.user_meetings) == 0

    @pytest.mark.asyncio
    async def test_meeting_events(self, manager, mock_meeting):
        """测试会议事件"""
        events = []
        
        @manager.on_meeting_created
        def on_created(room_id: str):
            events.append(('created', room_id))
            
        @manager.on_meeting_closed
        def on_closed(room_id: str):
            events.append(('closed', room_id))
            
        with patch('connect.jitsi.meeting_manager.JitsiMeeting', return_value=mock_meeting):
            room_id = await manager.create_meeting("test_host")
            await manager.close_meeting(room_id)
            
            assert ('created', room_id) in events
            assert ('closed', room_id) in events 
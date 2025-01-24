import pytest
import asyncio
from unittest.mock import Mock
from connect.jitsi.meeting import JitsiMeetingManager
from connect.jitsi.participant import JitsiParticipant

class TestJitsiMeetingManager:
    @pytest.fixture
    def setup_meeting(self):
        """初始化会议管理器"""
        config = {
            'max_participants': 16,
            'timeout': 30,
            'cleanup_interval': 60,
            'idle_timeout': 300
        }
        return JitsiMeetingManager(config)
        
    @pytest.mark.asyncio
    async def test_create_and_join(self, setup_meeting):
        """测试创建会议并加入多个参会者"""
        # 创建会议
        host_id = "host_1"
        meeting_id = await setup_meeting.create_meeting(host_id)
        
        # 加入多个参会者
        participants = ["user_1", "user_2", "user_3"]
        for user_id in participants:
            success = await setup_meeting.join_meeting(meeting_id, user_id)
            assert success
            
        room = setup_meeting.rooms[meeting_id]
        assert len(room.participants) == 4  # 主持人 + 3个参会者
        assert room.host_id == host_id

    @pytest.mark.asyncio
    async def test_leave_meeting(self, setup_meeting):
        """测试离开会议"""
        meeting_id = await setup_meeting.create_meeting("host_1")
        await setup_meeting.join_meeting(meeting_id, "user_1")
        
        success = await setup_meeting.leave_meeting(meeting_id, "user_1")
        assert success
        assert len(setup_meeting.rooms[meeting_id].participants) == 1

    @pytest.mark.asyncio
    async def test_idle_cleanup(self, setup_meeting):
        """测试会议空闲清理"""
        meeting_id = await setup_meeting.create_meeting("host_1")
        
        # 模拟会议空闲
        setup_meeting.rooms[meeting_id].last_active = 0
        
        await setup_meeting._cleanup_idle_rooms()
        assert meeting_id not in setup_meeting.rooms

    @pytest.mark.asyncio
    async def test_participant_permissions(self, setup_meeting):
        """测试参会者权限"""
        meeting_id = await setup_meeting.create_meeting("host_1")
        
        # 测试主持人权限
        assert await setup_meeting.has_permission("host_1", meeting_id, "kick_user")
        
        # 测试普通参会者权限
        await setup_meeting.join_meeting(meeting_id, "user_1")
        assert not await setup_meeting.has_permission("user_1", meeting_id, "kick_user")

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, setup_meeting):
        """测试并发操作"""
        meeting_id = await setup_meeting.create_meeting("host_1")
        
        # 并发加入测试
        join_tasks = [
            setup_meeting.join_meeting(meeting_id, f"user_{i}")
            for i in range(10)
        ]
        results = await asyncio.gather(*join_tasks)
        assert all(results[:15])  # 前15个应该成功
        assert not any(results[15:])  # 后面应该失败

    @pytest.mark.asyncio
    async def test_create_meeting(self, setup_meeting):
        """测试创建会议"""
        host_id = "test_host"
        meeting_id = await setup_meeting.create_meeting(host_id)
        assert meeting_id in setup_meeting.rooms
        assert setup_meeting.users[host_id] == meeting_id
        
    @pytest.mark.asyncio
    async def test_join_meeting(self, setup_meeting):
        """测试加入会议"""
        host_id = "test_host"
        user_id = "test_user"
        meeting_id = await setup_meeting.create_meeting(host_id)
        success = await setup_meeting.join_meeting(meeting_id, user_id)
        assert success
        assert len(setup_meeting.rooms[meeting_id].participants) == 2
        
    @pytest.mark.asyncio
    async def test_participant_limit(self, setup_meeting):
        """测试参会者数量限制"""
        meeting_id = await setup_meeting.create_meeting("host")
        # 尝试添加超过限制的参会者
        for i in range(20):
            success = await setup_meeting.join_meeting(
                meeting_id, f"user_{i}"
            )
            if i >= 15:  # 考虑主持人占用一个位置
                assert not success 
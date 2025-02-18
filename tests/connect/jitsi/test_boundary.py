import pytest
from connect.jitsi.meeting_manager import JitsiMeetingManager
from connect.jitsi.errors import MeetingError

@pytest.mark.asyncio
async def test_max_participants():
    """测试会议室最大人数限制"""
    manager = JitsiMeetingManager({'max_participants': 3})
    room_id = await manager.create_meeting("host")
    
    # 加入3个用户
    for i in range(3):
        await manager.join_meeting(room_id, f"user_{i}")
    
    # 尝试加入第四个用户应抛出异常
    with pytest.raises(MeetingError) as e:
        await manager.join_meeting(room_id, "user_4")
    assert "full" in str(e.value).lower()

@pytest.mark.asyncio
async def test_duplicate_join():
    """测试重复加入会议室"""
    manager = JitsiMeetingManager({})
    room_id = await manager.create_meeting("host")
    
    # 首次加入应成功
    assert await manager.join_meeting(room_id, "user1") is True
    
    # 重复加入应失败
    with pytest.raises(MeetingError) as e:
        await manager.join_meeting(room_id, "user1")
    assert "already exists" in str(e.value) 
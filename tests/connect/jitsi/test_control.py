import pytest
from connect.jitsi.client import JitsiMeetingController
import asyncio

@pytest.mark.asyncio
async def test_speaker_management():
    """测试发言权管理流程"""
    controller = JitsiMeetingController()
    
    # 测试申请发言
    await controller.request_speak("user1")
    await controller.request_speak("user2")
    assert controller._hand_raise_queue == ["user1", "user2"]
    
    # 测试授予发言权
    await controller.grant_speak("user1")
    assert controller._current_speaker == "user1"
    assert controller._hand_raise_queue == ["user2"]
    
    # 测试收回发言权
    await controller.revoke_speak()
    assert controller._current_speaker is None

@pytest.mark.asyncio
async def test_concurrent_requests():
    """测试并发请求处理"""
    controller = JitsiMeetingController()
    
    async def request(user_id):
        await controller.request_speak(user_id)
        
    # 模拟10个并发请求
    users = [f"user{i}" for i in range(10)]
    await asyncio.gather(*[request(uid) for uid in users])
    
    assert len(controller._hand_raise_queue) == 10
    assert len(set(controller._hand_raise_queue)) == 10 
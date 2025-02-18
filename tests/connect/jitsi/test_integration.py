import pytest
from unittest.mock import AsyncMock
from connect.jitsi.transport import JitsiTransport
from connect.jitsi.meeting_manager import JitsiMeetingManager
import asyncio

@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_meeting_flow():
    """测试完整会议流程"""
    # 初始化组件
    transport_config = {
        'buffer_size': 10,
        'batch_size': 2
    }
    transport = JitsiTransport(transport_config)
    manager = JitsiMeetingManager({})
    
    # Mock网络连接
    transport._jitsi = AsyncMock()
    manager.client = AsyncMock()
    
    # 创建会议室
    room_id = await manager.create_meeting("host_1")
    
    # 加入会议室
    await manager.join_meeting(room_id, "user_2")
    
    # 发送测试数据
    test_data = [b'data1', b'data2', b'data3']
    results = await transport.batch_send(test_data)
    
    # 验证结果
    assert len(results) == 3
    assert room_id in manager.active_rooms
    assert "user_2" in manager.active_rooms[room_id].participants 

@pytest.mark.integration
@pytest.mark.asyncio
async def test_large_meeting_scenario():
    """测试大型会议场景(50人)"""
    manager = JitsiMeetingManager({'max_participants': 50})
    transport = JitsiTransport({'batch_size': 20})
    
    # 创建会议室
    room_id = await manager.create_meeting("host_0")
    
    # 批量加入用户
    join_tasks = [
        manager.join_meeting(room_id, f"user_{i}") 
        for i in range(1, 50)
    ]
    await asyncio.gather(*join_tasks)
    
    # 验证参会人数
    assert len(manager.active_rooms[room_id].participants) == 50
    
    # 模拟数据广播
    messages = [b"pose_data"] * 1000
    results = await transport.batch_send(messages)
    
    # 验证传输统计
    stats = transport.get_stats()
    assert stats['sent'] >= 1000
    assert stats['errors'] == 0 
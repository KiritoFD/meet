import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import json
import time
import random
import psutil
import gc

from connect.jitsi.meeting_manager import JitsiMeetingManager
from connect.jitsi.meeting import JitsiMeeting
from connect.jitsi.participant import JitsiParticipant
from connect.jitsi.errors import MeetingError
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
            'cleanup_interval': 60,
            'heartbeat_interval': 30
        }
    }

@pytest.fixture
def mock_meeting():
    mock = AsyncMock()
    mock.room_id = "test_room"
    mock.host_id = "host_id"
    mock.get_participants = Mock(return_value=["host_id"])
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

@pytest.fixture
def meeting_config():
    return {
        'meeting': {
            'max_participants': 10,
            'idle_timeout': 300,  # 5分钟
            'cleanup_interval': 60,  # 1分钟
            'heartbeat_interval': 30  # 30秒
        }
    }

@pytest.fixture
def mock_participant():
    participant = AsyncMock(spec=JitsiParticipant)
    participant.id = "test_user"
    participant.is_connected = True
    return participant

@pytest.fixture
async def create_active_meeting(manager, mock_participant):
    """创建一个活跃的会议，包含参与者"""
    async def _create(room_id="test_room"):
        meeting = await manager.create_meeting(room_id)
        await manager.join_meeting(room_id, mock_participant)
        return meeting, room_id
    return _create

@pytest.fixture
def generate_random_activity():
    """生成随机的会议活动"""
    def _generate(count=10):
        activities = []
        for _ in range(count):
            activity_type = random.choice(['join', 'leave', 'message', 'status'])
            user_id = f"user_{random.randint(1, 100)}"
            activities.append((activity_type, user_id))
        return activities
    return _generate

class TestJitsiMeetingManager:
    @pytest.mark.asyncio
    async def test_create_meeting(self, manager, mock_meeting):
        """测试创建会议"""
        with patch('connect.jitsi.meeting_manager.JitsiMeeting', return_value=mock_meeting):
            room_id = await manager.create_meeting("host_id")
            
            assert room_id == mock_meeting.room_id
            assert room_id in manager.meetings
            assert manager.get_user_meeting("host_id") == room_id

    @pytest.mark.asyncio
    async def test_join_meeting(self, manager, mock_meeting, mock_participant):
        """测试加入会议"""
        with patch('connect.jitsi.meeting_manager.JitsiMeeting', return_value=mock_meeting):
            room_id = await manager.create_meeting("host_id")
            await manager.join_meeting(room_id, mock_participant)
            
            assert mock_participant in mock_meeting.participants
            assert mock_meeting.participant_count == 1
            assert manager.get_user_meeting("host_id") == room_id

    @pytest.mark.asyncio
    async def test_leave_meeting(self, manager, mock_meeting, mock_participant):
        """测试离开会议"""
        with patch('connect.jitsi.meeting_manager.JitsiMeeting', return_value=mock_meeting):
            room_id = await manager.create_meeting("host_id")
            await manager.join_meeting(room_id, mock_participant)
            
            await manager.leave_meeting(room_id, mock_participant.id)
            assert mock_participant not in mock_meeting.participants
            assert mock_meeting.participant_count == 0
            assert manager.get_user_meeting("host_id") is None

    @pytest.mark.asyncio
    async def test_close_meeting(self, manager, mock_meeting, mock_participant):
        """测试关闭会议"""
        with patch('connect.jitsi.meeting_manager.JitsiMeeting', return_value=mock_meeting):
            room_id = await manager.create_meeting("host_id")
            await manager.join_meeting(room_id, mock_participant)
            
            await manager.close_meeting(room_id)
            mock_meeting.close.assert_called_once()
            assert room_id not in manager.meetings
            assert not mock_meeting.is_active

    @pytest.mark.asyncio
    async def test_participant_limit(self, manager, mock_meeting):
        """测试参与者数量限制"""
        with patch('connect.jitsi.meeting_manager.JitsiMeeting', return_value=mock_meeting):
            room_id = await manager.create_meeting("host_id")
            
            # 模拟达到最大参与者数量
            mock_meeting.get_participants.return_value = ["host_id"] + [
                f"user_{i}" for i in range(manager.config['meeting']['max_participants'] - 1)
            ]
            
            # 尝试加入新用户应该失败
            success = await manager.join_meeting(room_id, "extra_user")
            assert not success

    @pytest.mark.asyncio
    async def test_idle_timeout(self, manager, mock_meeting):
        """测试空闲超时"""
        with patch('connect.jitsi.meeting_manager.JitsiMeeting', return_value=mock_meeting):
            room_id = await manager.create_meeting("host_id")
            
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
            room_id = await manager.create_meeting("host_id")
            
            # 非主持人不能关闭会议
            with pytest.raises(ValueError):
                await manager.close_meeting(room_id, "user1")
                
            # 主持人可以关闭会议
            await manager.close_meeting(room_id, "host_id")
            assert room_id not in manager.meetings

    @pytest.mark.asyncio
    async def test_meeting_stats(self, manager, mock_meeting):
        """测试会议统计"""
        with patch('connect.jitsi.meeting_manager.JitsiMeeting', return_value=mock_meeting):
            room_id = await manager.create_meeting("host_id")
            await manager.join_meeting(room_id, "user1")
            
            stats = manager.get_stats()
            assert stats['total_meetings'] == 1
            assert stats['active_meetings'] == 1
            assert stats['total_participants'] == 2

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, manager, mock_meeting):
        """测试并发操作"""
        with patch('connect.jitsi.meeting_manager.JitsiMeeting', return_value=mock_meeting):
            room_id = await manager.create_meeting("host_id")
            
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
            room_id = await manager.create_meeting("host_id")
            
            # 模拟加入失败
            mock_meeting.join.side_effect = JitsiError("Join failed")
            
            with pytest.raises(JitsiError):
                await manager.join_meeting(room_id, "user1")

    @pytest.mark.asyncio
    async def test_cleanup_on_close(self, manager, mock_meeting):
        """测试关闭时清理"""
        with patch('connect.jitsi.meeting_manager.JitsiMeeting', return_value=mock_meeting):
            room_id = await manager.create_meeting("host_id")
            await manager.close()
            
            mock_meeting.close.assert_called_once()
            assert len(manager.meetings) == 0
            assert len(manager.user_meetings) == 0

    @pytest.mark.asyncio
    async def test_meeting_events(self, manager, mock_meeting, mock_participant):
        """测试会议事件"""
        events = []
        
        @manager.on_meeting_created
        def on_created(room_id: str):
            events.append(('created', room_id))
            
        @manager.on_meeting_closed
        def on_closed(room_id: str):
            events.append(('closed', room_id))
            
        with patch('connect.jitsi.meeting_manager.JitsiMeeting', return_value=mock_meeting):
            room_id = await manager.create_meeting("host_id")
            await manager.join_meeting(room_id, mock_participant)
            await manager.leave_meeting(room_id, mock_participant.id)
            await manager.close_meeting(room_id)
            
            assert ('created', room_id) in events
            assert ('closed', room_id) in events

    @pytest.mark.asyncio
    async def test_cleanup_idle_meetings(self, manager):
        """测试清理空闲会议"""
        # 创建一个空闲会议
        room_id = "idle_room"
        meeting = await manager.create_meeting(room_id)
        meeting.last_activity = 0  # 设置为很久以前
        
        await manager.cleanup_idle_meetings()
        assert room_id not in manager.meetings

    @pytest.mark.asyncio
    async def test_heartbeat(self, manager, mock_meeting):
        """测试心跳机制"""
        with patch.object(manager, 'meetings', {'test_room': mock_meeting}):
            await manager.heartbeat()
            mock_meeting.check_participants.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_meeting_info(self, manager, mock_participant):
        """测试获取会议信息"""
        room_id = "test_room"
        meeting = await manager.create_meeting(room_id)
        await manager.join_meeting(room_id, mock_participant)
        
        info = await manager.get_meeting_info(room_id)
        assert info['room_id'] == room_id
        assert info['participant_count'] == 1
        assert info['is_active'] == True

    @pytest.mark.asyncio
    async def test_meeting_statistics(self, manager, mock_participant):
        """测试会议统计"""
        room_id = "test_room"
        meeting = await manager.create_meeting(room_id)
        
        # 记录一些活动
        await manager.join_meeting(room_id, mock_participant)
        await asyncio.sleep(0.1)
        await manager.leave_meeting(room_id, mock_participant.id)
        
        stats = await manager.get_meeting_statistics(room_id)
        assert stats['total_participants'] == 1
        assert stats['peak_participants'] == 1
        assert stats['average_duration'] > 0
        assert stats['total_messages'] >= 0

    @pytest.mark.asyncio
    async def test_meeting_state_transitions(self, manager, mock_participant):
        """测试会议状态转换"""
        room_id = "test_room"
        
        # 创建 -> 活跃
        meeting = await manager.create_meeting(room_id)
        assert meeting.state == "active"
        
        # 活跃 -> 暂停
        await manager.pause_meeting(room_id)
        assert meeting.state == "paused"
        
        # 暂停 -> 恢复
        await manager.resume_meeting(room_id)
        assert meeting.state == "active"
        
        # 活跃 -> 关闭
        await manager.close_meeting(room_id)
        assert meeting.state == "closed"

    @pytest.mark.asyncio
    async def test_meeting_data_consistency(self, manager, mock_participant):
        """测试会议数据一致性"""
        room_id = "test_room"
        meeting = await manager.create_meeting(room_id)
        
        # 检查内部数据结构一致性
        assert room_id in manager.meetings
        assert meeting.room_id in manager.active_meetings
        assert not meeting.room_id in manager.closed_meetings
        
        # 添加参与者后检查
        await manager.join_meeting(room_id, mock_participant)
        assert mock_participant.id in manager.user_meetings
        assert manager.user_meetings[mock_participant.id] == room_id
        
        # 关闭会议后检查
        await manager.close_meeting(room_id)
        assert room_id not in manager.meetings
        assert room_id not in manager.active_meetings
        assert mock_participant.id not in manager.user_meetings

    @pytest.mark.asyncio
    async def test_meeting_recovery(self, manager, mock_participant):
        """测试会议恢复机制"""
        room_id = "test_room"
        meeting = await manager.create_meeting(room_id)
        
        # 模拟异常情况
        await manager.join_meeting(room_id, mock_participant)
        meeting._connection.disconnect()  # 模拟连接断开
        
        # 测试自动恢复
        await manager.handle_connection_error(room_id)
        assert meeting.is_connected
        assert mock_participant in meeting.participants
        
        # 测试手动恢复
        await manager.force_reconnect(room_id)
        assert meeting.is_connected

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_performance(self, manager):
        """测试性能"""
        # 测试创建会议性能
        start_time = time.time()
        for i in range(100):
            await manager.create_meeting(f"room_{i}")
        create_time = time.time() - start_time
        assert create_time < 1.0  # 创建100个会议应该不超过1秒
        
        # 测试并发加入性能
        room_id = "perf_test_room"
        meeting = await manager.create_meeting(room_id)
        
        start_time = time.time()
        join_tasks = []
        for i in range(manager.config['meeting']['max_participants']):
            participant = AsyncMock(spec=JitsiParticipant)
            participant.id = f"user_{i}"
            join_tasks.append(manager.join_meeting(room_id, participant))
            
        await asyncio.gather(*join_tasks)
        join_time = time.time() - start_time
        assert join_time < 2.0  # 并发加入应该不超过2秒

    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_stress(self, manager):
        """压力测试"""
        # 模拟大量会议和参与者
        rooms = []
        participants = []
        
        # 创建多个会议
        for i in range(50):
            room_id = f"stress_room_{i}"
            meeting = await manager.create_meeting(room_id)
            rooms.append(room_id)
            
            # 每个会议加入多个参与者
            for j in range(5):
                participant = AsyncMock(spec=JitsiParticipant)
                participant.id = f"user_{i}_{j}"
                await manager.join_meeting(room_id, participant)
                participants.append(participant)
                
        # 进行密集操作
        operations = []
        for _ in range(1000):
            op_type = random.choice(['join', 'leave', 'status'])
            room_id = random.choice(rooms)
            participant = random.choice(participants)
            
            if op_type == 'join':
                operations.append(manager.join_meeting(room_id, participant))
            elif op_type == 'leave':
                operations.append(manager.leave_meeting(room_id, participant.id))
            else:
                operations.append(manager.update_participant_status(
                    room_id, participant.id, random.choice(['active', 'inactive'])
                ))
                
        # 并发执行所有操作
        await asyncio.gather(*operations, return_exceptions=True)
        
        # 验证系统稳定性
        assert len(manager.meetings) > 0
        for meeting in manager.meetings.values():
            assert meeting.is_active

    @pytest.mark.asyncio
    async def test_resource_management(self, manager):
        """测试资源管理"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # 创建和销毁大量会议
        for i in range(100):
            room_id = f"resource_room_{i}"
            meeting = await manager.create_meeting(room_id)
            await manager.close_meeting(room_id)
            
        # 强制垃圾回收
        gc.collect()
        
        # 检查内存使用
        current_memory = process.memory_info().rss
        memory_diff = current_memory - initial_memory
        assert memory_diff < 10 * 1024 * 1024  # 内存增长不应超过10MB

    @pytest.mark.asyncio
    async def test_meeting_cleanup_policy(self, manager):
        """测试会议清理策略"""
        # 创建不同状态的会议
        active_room = await manager.create_meeting("active_room")
        idle_room = await manager.create_meeting("idle_room")
        empty_room = await manager.create_meeting("empty_room")
        
        # 设置不同的状态
        idle_room.last_activity = time.time() - manager.config['meeting']['idle_timeout'] - 1
        empty_room.participant_count = 0
        
        # 运行清理
        await manager.cleanup_meetings()
        
        # 验证清理结果
        assert "active_room" in manager.meetings
        assert "idle_room" not in manager.meetings
        assert "empty_room" not in manager.meetings

    @pytest.mark.asyncio
    async def test_meeting_metrics_aggregation(self, manager):
        """测试会议指标聚合"""
        # 创建多个会议并生成活动
        meetings = []
        for i in range(3):
            room_id = f"room_{i}"
            meeting = await manager.create_meeting(room_id)
            meetings.append(meeting)
            
            # 模拟一些活动
            for j in range(2):
                participant = AsyncMock(spec=JitsiParticipant)
                participant.id = f"user_{i}_{j}"
                await manager.join_meeting(room_id, participant)
        
        # 获取聚合指标
        metrics = await manager.get_aggregated_metrics()
        
        # 验证聚合结果
        assert metrics['total_meetings'] == 3
        assert metrics['total_participants'] == 6
        assert metrics['average_participants_per_meeting'] == 2.0

    def test_config_validation(self):
        """测试配置验证"""
        # 测试必需字段
        with pytest.raises(ValueError):
            JitsiMeetingManager({})
            
        # 测试字段类型
        with pytest.raises(TypeError):
            JitsiMeetingManager({
                'meeting': {
                    'max_participants': "invalid",
                }
            })
            
        # 测试字段范围
        with pytest.raises(ValueError):
            JitsiMeetingManager({
                'meeting': {
                    'max_participants': -1,
                }
            })

    @pytest.mark.asyncio
    async def test_meeting_recording(self, manager, mock_participant):
        """测试会议录制功能"""
        room_id = "test_room"
        meeting = await manager.create_meeting(room_id)
        
        # 开始录制
        await manager.start_recording(room_id)
        assert meeting.is_recording
        
        # 暂停录制
        await manager.pause_recording(room_id)
        assert meeting.recording_paused
        
        # 恢复录制
        await manager.resume_recording(room_id)
        assert not meeting.recording_paused
        
        # 停止录制
        recording_info = await manager.stop_recording(room_id)
        assert not meeting.is_recording
        assert recording_info['duration'] > 0

    @pytest.mark.asyncio
    async def test_meeting_chat(self, manager, mock_participant):
        """测试会议聊天功能"""
        room_id = "test_room"
        meeting = await manager.create_meeting(room_id)
        await manager.join_meeting(room_id, mock_participant)
        
        # 发送消息
        message_id = await manager.send_message(room_id, mock_participant.id, "Hello")
        assert message_id is not None
        
        # 获取消息历史
        messages = await manager.get_chat_history(room_id)
        assert len(messages) == 1
        assert messages[0]['content'] == "Hello"
        
        # 删除消息
        await manager.delete_message(room_id, message_id)
        messages = await manager.get_chat_history(room_id)
        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_meeting_roles_and_permissions(self, manager, mock_participant):
        """测试会议角色和权限"""
        room_id = "test_room"
        meeting = await manager.create_meeting(room_id)
        
        # 测试角色分配
        await manager.assign_role(room_id, mock_participant.id, "moderator")
        assert await manager.get_role(room_id, mock_participant.id) == "moderator"
        
        # 测试权限检查
        assert await manager.check_permission(room_id, mock_participant.id, "kick_user")
        assert not await manager.check_permission(room_id, "regular_user", "kick_user")
        
        # 测试自定义权限
        await manager.set_custom_permissions(room_id, mock_participant.id, ["share_screen"])
        assert await manager.check_permission(room_id, mock_participant.id, "share_screen")

    @pytest.mark.asyncio
    async def test_meeting_breakout_rooms(self, manager, mock_participant):
        """测试分组会议室"""
        main_room = "main_room"
        meeting = await manager.create_meeting(main_room)
        
        # 创建分组会议室
        breakout_rooms = await manager.create_breakout_rooms(main_room, 3)
        assert len(breakout_rooms) == 3
        
        # 分配参与者到分组会议室
        await manager.assign_to_breakout_room(main_room, mock_participant.id, breakout_rooms[0])
        assert await manager.get_breakout_room(main_room, mock_participant.id) == breakout_rooms[0]
        
        # 关闭分组会议室
        await manager.close_breakout_rooms(main_room)
        assert len(await manager.get_active_breakout_rooms(main_room)) == 0

    @pytest.mark.asyncio
    async def test_meeting_state_persistence(self, manager):
        """测试会议状态持久化"""
        # 创建初始状态
        room_id = "test_room"
        meeting = await manager.create_meeting(room_id)
        
        # 保存状态
        state = await manager.save_state()
        
        # 模拟重启
        new_manager = JitsiMeetingManager(manager.config)
        await new_manager.restore_state(state)
        
        # 验证状态恢复
        restored_meeting = new_manager.meetings.get(room_id)
        assert restored_meeting is not None
        assert restored_meeting.config == meeting.config
        assert restored_meeting.state == meeting.state

    @pytest.mark.asyncio
    async def test_meeting_metrics_detailed(self, manager, mock_participant):
        """测试详细的会议指标"""
        room_id = "test_room"
        meeting = await manager.create_meeting(room_id)
        
        # 收集初始指标
        initial_metrics = await manager.get_detailed_metrics(room_id)
        
        # 生成一些活动
        await manager.join_meeting(room_id, mock_participant)
        await manager.send_message(room_id, mock_participant.id, "test")
        await manager.update_participant_status(room_id, mock_participant.id, "speaking")
        await manager.leave_meeting(room_id, mock_participant.id)
        
        # 获取更新后的指标
        metrics = await manager.get_detailed_metrics(room_id)
        
        # 验证各项指标
        assert metrics['network']['packet_loss'] < 0.01
        assert metrics['audio']['quality_score'] > 0.8
        assert metrics['video']['framerate'] > 24
        assert metrics['participants']['peak_count'] == 1 
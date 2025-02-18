import pytest
import asyncio
from unittest.mock import Mock
from connect.jitsi.transport import JitsiTransport
from connect.jitsi.meeting import JitsiMeetingManager
from pose.processor.jitsi_processor import JitsiPoseProcessor
from monitoring.jitsi.monitor import JitsiMonitor

class TestJitsiIntegration:
    @pytest.fixture
    async def setup_system(self):
        """初始化完整系统"""
        config = {
            'transport': {'buffer_size': 30},
            'meeting': {'max_participants': 16},
            'processing': {'compression_level': 6},
            'monitoring': {'metrics_interval': 5}
        }
        
        transport = JitsiTransport(config['transport'])
        meeting = JitsiMeetingManager(config['meeting'])
        processor = JitsiPoseProcessor(config['processing'])
        monitor = JitsiMonitor(config['monitoring'])
        
        event_bus = Mock()
        for component in [transport, meeting, processor, monitor]:
            component.set_event_bus(event_bus)
            
        yield {
            'transport': transport,
            'meeting': meeting,
            'processor': processor,
            'monitor': monitor,
            'event_bus': event_bus
        }
        
        # 清理资源
        await transport.close()
        
    @pytest.mark.asyncio
    async def test_end_to_end_flow(self, setup_system):
        """测试完整的端到端流程"""
        # 1. 创建会议
        meeting_id = await setup_system['meeting'].create_meeting("host_id")
        
        # 2. 处理姿态数据
        pose_data = self._create_test_pose()
        processed = setup_system['processor'].process(pose_data)
        
        # 3. 发送数据
        success = await setup_system['transport'].send(processed)
        assert success
        
        # 4. 检查监控指标
        metrics = setup_system['monitor'].metrics
        assert 'latency' in metrics
        assert metrics['latency'][-1] < 0.1  # 延迟应小于100ms 
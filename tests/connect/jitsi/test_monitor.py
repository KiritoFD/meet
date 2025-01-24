import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from connect.jitsi.monitor import JitsiMonitor
from connect.jitsi.metrics import (
    ConnectionMetrics, DataMetrics,
    MeetingMetrics, SystemMetrics
)

@pytest.fixture
def config():
    return {
        'monitor': {
            'window_size': 100,
            'sample_interval': 1.0,
            'alert_threshold': 0.9
        }
    }

@pytest.fixture
def monitor(config):
    return JitsiMonitor(config)

class TestJitsiMonitor:
    def test_record_metrics(self, monitor):
        metrics = ConnectionMetrics(
            latency=0.1,
            packet_loss=0.01,
            jitter=0.05,
            bandwidth=1000.0,
            state="connected"
        )
        monitor.record_connection_metrics(metrics)
        assert len(monitor._connection_metrics) == 1

    @pytest.mark.asyncio
    async def test_alert_handler(self, monitor):
        alerts = []
        def handler(alert):
            alerts.append(alert)
        
        monitor.add_alert_handler(handler)
        metrics = ConnectionMetrics(
            latency=0.1,
            packet_loss=0.95,  # 超过阈值
            jitter=0.05,
            bandwidth=1000.0,
            state="connected"
        )
        monitor.record_connection_metrics(metrics)
        assert len(alerts) > 0 
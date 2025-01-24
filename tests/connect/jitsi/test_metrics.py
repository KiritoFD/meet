import pytest
import time
from connect.jitsi.metrics import (
    ConnectionMetrics, DataMetrics, MeetingMetrics,
    SystemMetrics, PerformanceMetrics, MetricsAggregator
)

@pytest.fixture
def connection_metrics():
    return ConnectionMetrics(
        latency=0.1,
        packet_loss=0.01,
        jitter=0.05,
        bandwidth=1000.0,
        state="connected"
    )

@pytest.fixture
def data_metrics():
    return DataMetrics(
        bytes_sent=1000,
        bytes_received=2000,
        messages_sent=10,
        messages_received=20,
        compression_ratio=0.8
    )

@pytest.fixture
def meeting_metrics():
    return MeetingMetrics(
        room_id="test_room",
        participant_count=5,
        active_speakers=2,
        duration=300.0
    )

@pytest.fixture
def system_metrics():
    return SystemMetrics(
        cpu_usage=0.5,
        memory_usage=0.6,
        thread_count=10
    )

@pytest.fixture
def performance_metrics():
    return PerformanceMetrics(
        processing_time=0.05,
        queue_size=100,
        error_rate=0.001
    )

@pytest.fixture
def aggregator():
    return MetricsAggregator(window_size=5)

class TestMetrics:
    def test_connection_metrics_timestamp(self, connection_metrics):
        """测试连接指标时间戳"""
        assert connection_metrics.timestamp <= time.time()
        assert connection_metrics.timestamp > time.time() - 1

    def test_data_metrics_defaults(self):
        """测试数据指标默认值"""
        metrics = DataMetrics(
            bytes_sent=0,
            bytes_received=0,
            messages_sent=0,
            messages_received=0,
            compression_ratio=1.0
        )
        assert metrics.error_count == 0
        assert metrics.timestamp <= time.time()

    def test_meeting_metrics_peak_tracking(self):
        """测试会议峰值跟踪"""
        metrics = MeetingMetrics(
            room_id="test",
            participant_count=10,
            active_speakers=5,
            duration=0,
            peak_participants=8
        )
        assert metrics.peak_participants == 8
        assert metrics.total_messages == 0

    def test_system_metrics_optional_fields(self):
        """测试系统指标可选字段"""
        metrics = SystemMetrics(
            cpu_usage=0.5,
            memory_usage=0.6,
            thread_count=10,
            disk_io=100.0
        )
        assert metrics.disk_io == 100.0
        assert metrics.network_io is None

class TestMetricsAggregator:
    def test_add_metrics(self, aggregator, connection_metrics):
        """测试添加指标"""
        aggregator.add_metrics('connection', connection_metrics)
        assert len(aggregator.get_metrics('connection')) == 1

    def test_window_size(self, aggregator, connection_metrics):
        """测试窗口大小限制"""
        for _ in range(10):  # 超过窗口大小
            aggregator.add_metrics('connection', connection_metrics)
        assert len(aggregator.get_metrics('connection')) == 5  # window_size=5

    def test_get_latest(self, aggregator, connection_metrics, data_metrics):
        """测试获取最新指标"""
        aggregator.add_metrics('connection', connection_metrics)
        latest = aggregator.get_latest('connection')
        assert latest == connection_metrics
        assert aggregator.get_latest('unknown') is None

    def test_get_average(self, aggregator):
        """测试计算平均值"""
        metrics = [
            ConnectionMetrics(latency=0.1 * i, packet_loss=0.01, 
                            jitter=0.05, bandwidth=1000.0, state="connected")
            for i in range(1, 4)
        ]
        for m in metrics:
            aggregator.add_metrics('connection', m)
            
        avg_latency = aggregator.get_average('connection', 'latency')
        assert avg_latency == pytest.approx(0.2)  # (0.1 + 0.2 + 0.3) / 3

    def test_get_summary(self, aggregator, connection_metrics, data_metrics):
        """测试获取汇总信息"""
        aggregator.add_metrics('connection', connection_metrics)
        aggregator.add_metrics('data', data_metrics)
        
        summary = aggregator.get_summary()
        assert 'connection' in summary
        assert 'data' in summary
        assert summary['connection']['count'] == 1
        assert summary['data']['latest'] == data_metrics

    def test_clear_metrics(self, aggregator, connection_metrics, data_metrics):
        """测试清除指标"""
        aggregator.add_metrics('connection', connection_metrics)
        aggregator.add_metrics('data', data_metrics)
        
        # 清除特定类型
        aggregator.clear('connection')
        assert len(aggregator.get_metrics('connection')) == 0
        assert len(aggregator.get_metrics('data')) == 1
        
        # 清除所有
        aggregator.clear()
        assert len(aggregator.get_metrics('data')) == 0

    def test_invalid_metric_type(self, aggregator, connection_metrics):
        """测试无效的指标类型"""
        with pytest.raises(ValueError):
            aggregator.add_metrics('invalid_type', connection_metrics)

    def test_empty_metrics_handling(self, aggregator):
        """测试空指标处理"""
        assert aggregator.get_latest('connection') is None
        assert aggregator.get_average('connection', 'latency') is None
        summary = aggregator.get_summary()
        assert 'connection' not in summary 
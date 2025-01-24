import pytest
import time
import random
import asyncio
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

    @pytest.mark.asyncio
    async def test_metrics_sampling(self, metrics_aggregator):
        """测试指标采样"""
        # 生成测试数据
        for i in range(100):
            metrics = {
                'timestamp': time.time() + i,
                'value': random.random() * 100
            }
            metrics_aggregator.add_sample('test_metric', metrics)
            
        # 测试不同采样方法
        # 平均采样
        avg_samples = metrics_aggregator.get_samples('test_metric', 
                                                   method='average', 
                                                   window=10)
        assert len(avg_samples) == 10
        
        # 最大值采样
        max_samples = metrics_aggregator.get_samples('test_metric',
                                                   method='max',
                                                   window=10)
        assert len(max_samples) == 10
        assert all(s['value'] >= avg_samples[i]['value'] 
                  for i, s in enumerate(max_samples))

    @pytest.mark.asyncio
    async def test_metrics_aggregation_methods(self, metrics_aggregator):
        """测试指标聚合方法"""
        # 添加测试数据
        values = [1, 2, 3, 4, 5]
        for v in values:
            metrics_aggregator.add_value('test_metric', v)
            
        # 测试不同聚合方法
        assert metrics_aggregator.get_average('test_metric') == 3.0
        assert metrics_aggregator.get_max('test_metric') == 5
        assert metrics_aggregator.get_min('test_metric') == 1
        assert metrics_aggregator.get_median('test_metric') == 3
        assert metrics_aggregator.get_percentile('test_metric', 90) == 5
        assert metrics_aggregator.get_stddev('test_metric') > 0

    @pytest.mark.asyncio
    async def test_metrics_windowing(self, metrics_aggregator):
        """测试指标时间窗口"""
        current_time = time.time()
        
        # 添加跨越多个时间窗口的数据
        windows = [
            (current_time - 3600, 1),  # 1小时前
            (current_time - 1800, 2),  # 30分钟前
            (current_time - 300, 3),   # 5分钟前
            (current_time, 4)          # 当前
        ]
        
        for timestamp, value in windows:
            metrics_aggregator.add_sample('test_metric', {
                'timestamp': timestamp,
                'value': value
            })
            
        # 测试不同时间窗口的聚合
        assert metrics_aggregator.get_average('test_metric', window=3600) == 2.5
        assert metrics_aggregator.get_average('test_metric', window=1800) == 3.0
        assert metrics_aggregator.get_average('test_metric', window=300) == 3.5

    @pytest.mark.asyncio
    async def test_metrics_validation(self, metrics_aggregator):
        """测试指标验证"""
        # 测试无效指标名
        with pytest.raises(ValueError):
            metrics_aggregator.add_value('', 1)
            
        # 测试无效值
        with pytest.raises(ValueError):
            metrics_aggregator.add_value('test', float('inf'))
            
        # 测试无效时间戳
        with pytest.raises(ValueError):
            metrics_aggregator.add_sample('test', {
                'timestamp': -1,
                'value': 1
            })
            
        # 测试无效窗口大小
        with pytest.raises(ValueError):
            metrics_aggregator.get_average('test', window=-1)

    @pytest.mark.asyncio
    async def test_metrics_persistence(self, metrics_aggregator):
        """测试指标持久化"""
        # 添加测试数据
        metrics_aggregator.add_value('test', 1)
        
        # 保存状态
        state = metrics_aggregator.save_state()
        
        # 创建新的聚合器并恢复状态
        new_aggregator = MetricsAggregator()
        new_aggregator.restore_state(state)
        
        # 验证状态恢复
        assert new_aggregator.get_average('test') == 1
        assert new_aggregator.get_metrics_names() == {'test'}

    @pytest.mark.asyncio
    async def test_metrics_cleanup(self, metrics_aggregator):
        """测试指标清理"""
        current_time = time.time()
        
        # 添加过期数据
        metrics_aggregator.add_sample('test', {
            'timestamp': current_time - 7200,  # 2小时前
            'value': 1
        })
        
        # 添加有效数据
        metrics_aggregator.add_sample('test', {
            'timestamp': current_time,
            'value': 2
        })
        
        # 执行清理
        metrics_aggregator.cleanup(max_age=3600)  # 1小时
        
        # 验证结果
        assert metrics_aggregator.get_average('test') == 2

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_metrics_performance(self, metrics_aggregator):
        """测试指标性能"""
        # 测试大量数据添加性能
        start_time = time.time()
        for i in range(10000):
            metrics_aggregator.add_value('test', i)
        add_time = time.time() - start_time
        assert add_time < 1.0  # 添加10000个值应在1秒内完成
        
        # 测试聚合计算性能
        start_time = time.time()
        for _ in range(100):
            metrics_aggregator.get_average('test')
            metrics_aggregator.get_percentile('test', 95)
        calc_time = time.time() - start_time
        assert calc_time < 0.1  # 100次计算应在0.1秒内完成
        
        # 测试内存使用
        import psutil
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        assert memory_usage < 100  # 内存使用不应超过100MB

    @pytest.mark.asyncio
    async def test_metrics_correlation(self, metrics_aggregator):
        """测试指标相关性分析"""
        # 添加相关的指标数据
        for i in range(100):
            metrics_aggregator.add_sample('cpu_usage', {
                'timestamp': time.time() + i,
                'value': i * 0.01
            })
            metrics_aggregator.add_sample('latency', {
                'timestamp': time.time() + i,
                'value': i * 0.005 + random.random() * 0.1
            })
        
        # 计算相关性
        correlation = metrics_aggregator.calculate_correlation(
            'cpu_usage', 'latency', window=60
        )
        assert 0.5 < correlation < 1.0  # 应该有正相关性

    @pytest.mark.asyncio
    async def test_metrics_anomaly_detection(self, metrics_aggregator):
        """测试异常检测"""
        # 添加正常数据
        for i in range(100):
            metrics_aggregator.add_value('test', 100 + random.random() * 10)
            
        # 添加异常值
        metrics_aggregator.add_value('test', 500)  # 明显偏高
        metrics_aggregator.add_value('test', 0)    # 明显偏低
        
        # 检测异常
        anomalies = metrics_aggregator.detect_anomalies('test', 
                                                      window=50,
                                                      threshold=3)  # 3个标准差
        assert len(anomalies) == 2
        assert all(a['is_anomaly'] for a in anomalies)

    @pytest.mark.asyncio
    async def test_metrics_forecasting(self, metrics_aggregator):
        """测试指标预测"""
        # 添加历史数据
        for i in range(100):
            metrics_aggregator.add_sample('test', {
                'timestamp': time.time() + i * 60,  # 每分钟一个样本
                'value': 100 + i + random.random() * 10  # 线性增长加噪声
            })
        
        # 预测未来值
        forecast = metrics_aggregator.forecast('test', 
                                            horizon=5,    # 预测未来5个点
                                            window=60)    # 使用60个历史点
        
        assert len(forecast) == 5
        assert all(150 < f['value'] < 250 for f in forecast)  # 合理的预测范围

    @pytest.mark.asyncio
    async def test_metrics_error_handling(self, metrics_aggregator):
        """测试错误处理"""
        # 测试并发访问
        async def concurrent_add():
            for i in range(1000):
                metrics_aggregator.add_value('concurrent', i)
                
        tasks = [concurrent_add() for _ in range(10)]
        await asyncio.gather(*tasks)
        
        # 验证数据完整性
        assert metrics_aggregator.get_count('concurrent') == 10000
        
        # 测试数据类型错误
        with pytest.raises(TypeError):
            metrics_aggregator.add_value('test', "not a number")
            
        # 测试指标名冲突
        metrics_aggregator.add_value('conflict', 1)
        with pytest.raises(ValueError):
            metrics_aggregator.add_metrics('conflict', 
                                         ConnectionMetrics(latency=0.1))

    @pytest.mark.asyncio
    async def test_metrics_edge_cases(self, metrics_aggregator):
        """测试边界情况"""
        # 测试空窗口
        metrics_aggregator.add_value('test', 1)
        assert metrics_aggregator.get_average('test', window=0) == 1
        
        # 测试单个值的统计
        assert metrics_aggregator.get_stddev('test') == 0
        assert metrics_aggregator.get_median('test') == 1
        
        # 测试大量重复值
        for _ in range(1000):
            metrics_aggregator.add_value('constant', 5)
        assert metrics_aggregator.get_stddev('constant') == 0
        assert metrics_aggregator.get_percentile('constant', 99) == 5

class TestBaseMetrics:
    def test_connection_metrics_validation(self, connection_metrics):
        """测试连接指标验证"""
        # 测试有效范围
        assert 0 <= connection_metrics.packet_loss <= 1
        assert connection_metrics.latency >= 0
        assert connection_metrics.bandwidth > 0
        
        # 测试无效值
        with pytest.raises(ValueError):
            ConnectionMetrics(latency=-1)
        with pytest.raises(ValueError):
            ConnectionMetrics(packet_loss=2)
        with pytest.raises(ValueError):
            ConnectionMetrics(bandwidth=-100)

    def test_metrics_serialization(self, connection_metrics, data_metrics):
        """测试指标序列化"""
        # 测试转换为字典
        conn_dict = connection_metrics.to_dict()
        assert conn_dict['latency'] == 0.1
        assert conn_dict['state'] == "connected"
        
        # 测试从字典创建
        new_metrics = ConnectionMetrics.from_dict(conn_dict)
        assert new_metrics.latency == connection_metrics.latency
        assert new_metrics.state == connection_metrics.state

    def test_metrics_comparison(self, connection_metrics):
        """测试指标比较"""
        better_metrics = ConnectionMetrics(
            latency=0.05,  # 更低延迟
            packet_loss=0.005,  # 更低丢包
            jitter=0.02,  # 更低抖动
            bandwidth=2000.0,  # 更高带宽
            state="connected"
        )
        
        assert better_metrics > connection_metrics
        assert connection_metrics < better_metrics
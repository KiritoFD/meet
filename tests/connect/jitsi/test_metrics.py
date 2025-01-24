import pytest
import time
import random
import asyncio
import math
import os
import psutil
from unittest.mock import patch, AsyncMock
from connect.jitsi.metrics import (
    ConnectionMetrics, DataMetrics, MeetingMetrics,
    SystemMetrics, PerformanceMetrics, MetricsAggregator, QualityMetrics,
    MetricsError, PerformanceError  # 添加错误类型导入
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
def metrics_aggregator():
    """创建指标聚合器"""
    aggregator = MetricsAggregator(
        window_size=3600,
        storage_path="/tmp/metrics",
        max_points=10000,
        cleanup_interval=300
    )
    try:
        yield aggregator
    finally:
        # 确保在测试结束后清理资源
        try:
            aggregator.clear_all()
            if os.path.exists("/tmp/metrics"):
                import shutil
                shutil.rmtree("/tmp/metrics")
        except Exception as e:
            print(f"Warning: Failed to cleanup test data: {e}")
            # 不抛出异常，避免掩盖测试失败

@pytest.fixture
def quality_metrics():
    """创建质量指标"""
    return QualityMetrics(
        video_quality=0.85,
        audio_quality=0.9,
        screen_quality=0.95,
        frame_rate=30,
        resolution="1080p"
    )

@pytest.fixture
async def sample_data():
    """生成样本数据"""
    return {
        'periodic': [(time.time() + i * 60, math.sin(i * math.pi / 12)) 
                    for i in range(100)],
        'linear': [(time.time() + i * 60, i * 0.1) 
                  for i in range(100)],
        'random': [(time.time() + i * 60, random.random()) 
                  for i in range(100)]
    }

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

    def test_metrics_thread_safety(self, metrics_aggregator):
        """测试线程安全性"""
        import threading
        
        def worker():
            for i in range(1000):
                metrics_aggregator.add_value('concurrent', i)
                
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
            
        assert metrics_aggregator.get_count('concurrent') == 10000

    def test_metrics_value_constraints(self, metrics_aggregator):
        """测试值约束"""
        # 测试最大/最小值约束
        metrics_aggregator.set_constraints('test', min_value=0, max_value=100)
        
        metrics_aggregator.add_value('test', 50)  # 正常值
        with pytest.raises(ValueError):
            metrics_aggregator.add_value('test', -1)  # 低于最小值
        with pytest.raises(ValueError):
            metrics_aggregator.add_value('test', 101)  # 高于最大值
            
        # 测试自定义验证器
        def validate_even(value):
            return value % 2 == 0
            
        metrics_aggregator.set_validator('even_numbers', validate_even)
        metrics_aggregator.add_value('even_numbers', 2)  # 有效值
        with pytest.raises(ValueError):
            metrics_aggregator.add_value('even_numbers', 3)  # 无效值

    def test_metrics_error_handling_edge_cases(self, metrics_aggregator):
        """测试边缘情况的错误处理"""
        # 测试重复初始化
        with pytest.raises(RuntimeError):
            metrics_aggregator.initialize()
            
        # 测试无效的聚合方法
        with pytest.raises(ValueError):
            metrics_aggregator.aggregate('test', method='invalid')
            
        # 测试并发事务冲突
        metrics_aggregator.start_transaction()
        with pytest.raises(RuntimeError):
            metrics_aggregator.start_transaction()  # 嵌套事务
            
        # 测试未开始事务就提交
        with pytest.raises(RuntimeError):
            metrics_aggregator.commit_transaction()
            
        # 测试数据类型不一致
        metrics_aggregator.add_value('test', 1)
        with pytest.raises(TypeError):
            metrics_aggregator.add_value('test', 'string')  # 类型不匹配

class TestMetricsAggregator:
    def test_add_metrics(self, metrics_aggregator, connection_metrics):
        """测试添加指标"""
        metrics_aggregator.add_metrics('connection', connection_metrics)
        assert len(metrics_aggregator.get_metrics('connection')) == 1

    def test_window_size(self, metrics_aggregator, connection_metrics):
        """测试窗口大小限制"""
        metrics_aggregator.set_window_size(5)  # 显式设置窗口大小
        for _ in range(10):
            metrics_aggregator.add_metrics('connection', connection_metrics)
        assert len(metrics_aggregator.get_metrics('connection')) == 5

    def test_get_latest(self, metrics_aggregator, connection_metrics, data_metrics):
        """测试获取最新指标"""
        metrics_aggregator.add_metrics('connection', connection_metrics)
        latest = metrics_aggregator.get_latest('connection')
        assert latest == connection_metrics
        assert metrics_aggregator.get_latest('unknown') is None

    def test_get_average(self, metrics_aggregator):
        """测试计算平均值"""
        metrics = [
            ConnectionMetrics(latency=0.1 * i, packet_loss=0.01, 
                            jitter=0.05, bandwidth=1000.0, state="connected")
            for i in range(1, 4)
        ]
        for m in metrics:
            metrics_aggregator.add_metrics('connection', m)
            
        avg_latency = metrics_aggregator.get_average('connection', 'latency')
        assert avg_latency == pytest.approx(0.2)  # (0.1 + 0.2 + 0.3) / 3

    def test_get_summary(self, metrics_aggregator, connection_metrics, data_metrics):
        """测试获取汇总信息"""
        metrics_aggregator.add_metrics('connection', connection_metrics)
        metrics_aggregator.add_metrics('data', data_metrics)
        
        summary = metrics_aggregator.get_summary()
        assert 'connection' in summary
        assert 'data' in summary
        assert summary['connection']['count'] == 1
        assert summary['data']['latest'] == data_metrics

    def test_clear_metrics(self, metrics_aggregator, connection_metrics, data_metrics):
        """测试清除指标"""
        metrics_aggregator.add_metrics('connection', connection_metrics)
        metrics_aggregator.add_metrics('data', data_metrics)
        
        # 清除特定类型
        metrics_aggregator.clear('connection')
        assert len(metrics_aggregator.get_metrics('connection')) == 0
        assert len(metrics_aggregator.get_metrics('data')) == 1
        
        # 清除所有
        metrics_aggregator.clear()
        assert len(metrics_aggregator.get_metrics('data')) == 0

    def test_invalid_metric_type(self, metrics_aggregator, connection_metrics):
        """测试无效的指标类型"""
        with pytest.raises(ValueError):
            metrics_aggregator.add_metrics('invalid_type', connection_metrics)

    def test_empty_metrics_handling(self, metrics_aggregator):
        """测试空指标处理"""
        assert metrics_aggregator.get_latest('connection') is None
        assert metrics_aggregator.get_average('connection', 'latency') is None
        summary = metrics_aggregator.get_summary()
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

    def test_metrics_aggregation_methods(self, metrics_aggregator):
        """测试不同的聚合方法"""
        values = [1, 2, 3, 4, 5]
        for v in values:
            metrics_aggregator.add_value('test', v)
            
        # 测试各种聚合方法
        assert metrics_aggregator.aggregate('test', method='sum') == 15
        assert metrics_aggregator.aggregate('test', method='min') == 1
        assert metrics_aggregator.aggregate('test', method='max') == 5
        assert metrics_aggregator.aggregate('test', method='avg') == 3
        assert metrics_aggregator.aggregate('test', method='median') == 3
        assert metrics_aggregator.aggregate('test', method='percentile90') == 4.6
        
        # 测试无效的聚合方法
        with pytest.raises(ValueError):
            metrics_aggregator.aggregate('test', method='invalid')

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

    @pytest.mark.asyncio
    async def test_metrics_time_series(self, metrics_aggregator):
        """测试时间序列分析"""
        # 生成周期性数据
        for i in range(100):
            value = 10 + 5 * math.sin(i * math.pi / 12)  # 24点周期
            metrics_aggregator.add_sample('periodic', {
                'timestamp': time.time() + i * 3600,  # 每小时一个点
                'value': value
            })
            
        # 测试趋势分析
        trend = metrics_aggregator.analyze_trend('periodic', window=24)
        assert 'slope' in trend
        assert 'r_squared' in trend
        
        # 测试季节性检测
        seasonality = metrics_aggregator.detect_seasonality('periodic')
        assert seasonality['period'] == 24  # 应检测出24小时周期
        
        # 测试移动平均
        ma = metrics_aggregator.moving_average('periodic', window=12)
        assert len(ma) > 0
        assert all(8 <= x['value'] <= 12 for x in ma)  # 应该围绕10上下波动

    @pytest.mark.asyncio
    async def test_metrics_error_handling_comprehensive(self, metrics_aggregator):
        """全面的错误处理测试"""
        # 测试数据一致性
        with pytest.raises(ValueError):
            # 时间戳倒退
            metrics_aggregator.add_sample('test', {
                'timestamp': time.time(),
                'value': 1
            })
            metrics_aggregator.add_sample('test', {
                'timestamp': time.time() - 3600,
                'value': 2
            })
            
        # 测试数据类型转换
        metrics_aggregator.add_value('test', '123')  # 应该自动转换为数字
        assert isinstance(metrics_aggregator.get_latest('test'), float)
        
        # 测试空值处理
        metrics_aggregator.add_value('test', None)  # 应该忽略
        assert metrics_aggregator.get_count('test') == 1
        
        # 测试并发写入冲突
        async def conflicting_writes():
            metrics_aggregator.start_transaction()
            await asyncio.sleep(0.1)
            metrics_aggregator.add_value('conflict', 1)
            metrics_aggregator.commit_transaction()
            
        with pytest.raises(RuntimeError):
            await asyncio.gather(
                conflicting_writes(),
                conflicting_writes()
            )

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_metrics_optimization(self, metrics_aggregator):
        """测试性能优化"""
        # 测试缓存机制
        start_time = time.time()
        for _ in range(100):
            metrics_aggregator.get_average('test')  # 第一次计算后应该使用缓存
        cache_time = time.time() - start_time
        assert cache_time < 0.01  # 缓存访问应该很快
        
        # 测试批量操作性能
        values = [(time.time() + i, i) for i in range(1000)]
        start_time = time.time()
        metrics_aggregator.add_samples_batch('test', values)
        batch_time = time.time() - start_time
        
        # 对比单个添加的时间
        start_time = time.time()
        for t, v in values:
            metrics_aggregator.add_sample('test2', {'timestamp': t, 'value': v})
        single_time = time.time() - start_time
        
        assert batch_time < single_time / 2  # 批量操作应该至少快2倍

    @pytest.mark.asyncio
    async def test_metrics_aggregator_advanced(self, metrics_aggregator):
        """测试聚合器高级功能"""
        # 测试分组统计
        for i in range(100):
            metrics_aggregator.add_sample('test', {
                'timestamp': time.time(),
                'value': i,
                'category': 'A' if i < 50 else 'B'
            })
            
        stats_by_category = metrics_aggregator.group_by('test', 'category')
        assert stats_by_category['A']['average'] == 24.5  # 0-49的平均值
        assert stats_by_category['B']['average'] == 74.5  # 50-99的平均值
        
        # 测试复合指标计算
        metrics_aggregator.add_derived_metric(
            'efficiency',
            lambda m: m.get_value('throughput') / m.get_value('cpu_usage')
        )
        
        metrics_aggregator.add_value('throughput', 100)
        metrics_aggregator.add_value('cpu_usage', 0.5)
        assert metrics_aggregator.get_value('efficiency') == 200

    @pytest.mark.asyncio
    async def test_metrics_data_retention(self, metrics_aggregator):
        """测试数据保留策略"""
        # 添加数据
        for i in range(1000):
            metrics_aggregator.add_sample('test', {
                'timestamp': time.time() - i * 3600,  # 每小时一个点
                'value': i
            })
            
        # 测试数据压缩
        metrics_aggregator.compress_data('test', max_points=100)
        assert metrics_aggregator.get_count('test') <= 100
        
        # 测试数据归档
        archive_path = metrics_aggregator.archive_data('test', 
                                                     older_than=24*3600)
        assert os.path.exists(archive_path)
        
        # 测试数据恢复
        metrics_aggregator.restore_from_archive(archive_path)
        assert metrics_aggregator.get_count('test') > 0

    @pytest.mark.asyncio
    async def test_metrics_fault_tolerance(self, metrics_aggregator):
        """测试容错机制"""
        # 模拟存储失败
        with patch.object(metrics_aggregator, '_save_to_storage',
                         side_effect=IOError):
            # 应该继续工作并使用内存缓存
            metrics_aggregator.add_value('test', 1)
            assert metrics_aggregator.get_latest('test') == 1
            
        # 模拟部分数据损坏
        metrics_aggregator.add_value('test', 1)
        metrics_aggregator.add_value('test', 'invalid')  # 应该被忽略
        metrics_aggregator.add_value('test', 2)
        
        values = metrics_aggregator.get_values('test')
        assert len(values) == 2
        assert all(isinstance(v, (int, float)) for v in values)

    @pytest.mark.asyncio
    async def test_metrics_consistency(self, metrics_aggregator):
        """测试数据一致性"""
        # 测试原子性
        async def concurrent_updates():
            async with metrics_aggregator.transaction():
                metrics_aggregator.add_value('atomic', 1)
                await asyncio.sleep(0.1)
                metrics_aggregator.add_value('atomic', 2)
                
        await asyncio.gather(
            concurrent_updates(),
            concurrent_updates()
        )
        
        # 验证事务完整性
        values = set(metrics_aggregator.get_values('atomic'))
        assert values == {1, 2}  # 每个事务的值都应该完整保存

    @pytest.mark.asyncio
    async def test_metrics_data_processing(self, metrics_aggregator, sample_data):
        """测试数据处理功能"""
        # 测试数据导入
        for metric_type, data in sample_data.items():
            metrics_aggregator.import_data(metric_type, data)
            
        # 测试数据导出
        for metric_type in sample_data:
            exported = metrics_aggregator.export_data(metric_type)
            assert len(exported) == len(sample_data[metric_type])
            
        # 测试数据转换
        transformed = metrics_aggregator.transform_data(
            'periodic',
            transform_fn=lambda x: x * 2
        )
        assert all(v == 2 * original 
                  for (_, v), (_, original) in zip(transformed, sample_data['periodic']))
        
        # 测试数据过滤
        filtered = metrics_aggregator.filter_data(
            'linear',
            filter_fn=lambda x: x > 5
        )
        assert all(v > 5 for _, v in filtered)

    @pytest.mark.asyncio
    async def test_metrics_recovery(self, metrics_aggregator):
        """测试错误恢复"""
        # 模拟数据损坏
        metrics_aggregator.add_value('test', 1)
        metrics_aggregator._data['test'].append('corrupted')
        
        # 测试自动修复
        metrics_aggregator.repair_data('test')
        assert all(isinstance(v, (int, float)) 
                  for v in metrics_aggregator.get_values('test'))
        
        # 测试数据备份和恢复
        backup = metrics_aggregator.create_backup()
        metrics_aggregator.clear_all()
        metrics_aggregator.restore_from_backup(backup)
        assert metrics_aggregator.get_count('test') > 0

    @pytest.mark.benchmark
    def test_metrics_memory_optimization(self, metrics_aggregator):
        """测试内存优化"""
        process = psutil.Process()
        
        # 测试内存自动清理
        initial_memory = process.memory_info().rss
        for i in range(100000):
            metrics_aggregator.add_value(f'test_{i}', i)
        
        # 触发清理
        metrics_aggregator.optimize_memory(target_size_mb=50)
        
        final_memory = process.memory_info().rss
        memory_increase_mb = (final_memory - initial_memory) / 1024 / 1024
        assert memory_increase_mb < 50  # 内存增长应该受控

    @pytest.mark.asyncio
    async def test_metrics_concurrent_access(self, metrics_aggregator):
        """测试并发访问控制"""
        async def concurrent_operation(op_id):
            async with metrics_aggregator.async_transaction():
                metrics_aggregator.add_value(f'concurrent_{op_id}', op_id)
                await asyncio.sleep(0.01)
                value = metrics_aggregator.get_latest(f'concurrent_{op_id}')
                assert value == op_id

        # 执行多个并发操作
        tasks = [concurrent_operation(i) for i in range(10)]
        await asyncio.gather(*tasks)

    @pytest.mark.asyncio
    async def test_metrics_rate_limiting(self, metrics_aggregator):
        """测试速率限制"""
        metrics_aggregator.set_rate_limit(100)  # 每秒100个样本
        
        start_time = time.time()
        for i in range(200):
            metrics_aggregator.add_value('rate_limited', i)
        
        elapsed = time.time() - start_time
        assert elapsed >= 2.0  # 应该至少需要2秒

    def test_metrics_configuration(self):
        """测试配置验证"""
        # 测试无效配置
        with pytest.raises(ValueError):
            MetricsAggregator(window_size=-1)
        
        with pytest.raises(ValueError):
            MetricsAggregator(max_points=0)
            
        # 测试有效配置
        aggregator = MetricsAggregator(
            window_size=3600,
            max_points=1000,
            cleanup_interval=300
        )
        assert aggregator.window_size == 3600
        assert aggregator.max_points == 1000

    def test_metrics_resource_management(self, metrics_aggregator):
        """测试资源管理"""
        # 测试自动清理
        for i in range(10000):
            metrics_aggregator.add_value('resource_test', i)
        
        # 验证内存使用
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # 触发清理
        metrics_aggregator.cleanup()
        
        # 验证资源释放
        assert metrics_aggregator.get_count('resource_test') <= metrics_aggregator.max_points
        assert process.memory_info().rss / 1024 / 1024 <= memory_mb

    def test_metrics_errors(self):
        """测试指标错误类型"""
        # 测试基本错误
        with pytest.raises(MetricsError) as exc_info:
            raise MetricsError("Test error")
        assert str(exc_info.value) == "Test error"
        
        # 测试带错误码的错误
        error = MetricsError("Test error", code=1001)
        assert error.code == 1001
        assert str(error) == "Test error (code: 1001)"

    @pytest.mark.asyncio
    async def test_metrics_async_context(self, metrics_aggregator):
        """测试异步上下文管理器"""
        async with metrics_aggregator.async_session() as session:
            await session.add_value('test', 1)
            assert await session.get_latest('test') == 1
            
        # 验证会话已正确关闭
        assert session.is_closed
        
        # 验证数据已持久化
        assert metrics_aggregator.get_latest('test') == 1

    @pytest.mark.asyncio
    async def test_metrics_batch_operations_errors(self, metrics_aggregator):
        """测试批量操作错误处理"""
        # 测试部分失败的批量添加
        values = [
            (time.time(), 1),
            (time.time(), 'invalid'),  # 无效值
            (time.time(), 2)
        ]
        
        results = metrics_aggregator.add_samples_batch(
            'test',
            values,
            continue_on_error=True
        )
        
        assert len(results.succeeded) == 2
        assert len(results.failed) == 1
        assert isinstance(results.failed[0].error, TypeError)
        
        # 验证有效数据已添加
        assert metrics_aggregator.get_count('test') == 2

    def test_metrics_initialization(self):
        """测试指标初始化"""
        # 测试默认值
        metrics = MetricsAggregator()
        assert metrics.window_size == 3600  # 默认1小时
        assert metrics.max_points == 10000
        assert metrics.cleanup_interval == 300
        
        # 测试自定义配置
        custom_metrics = MetricsAggregator(
            window_size=7200,
            max_points=20000,
            cleanup_interval=600,
            storage_path="/custom/path"
        )
        assert custom_metrics.window_size == 7200
        assert custom_metrics.max_points == 20000
        assert custom_metrics.cleanup_interval == 600
        assert custom_metrics.storage_path == "/custom/path"

    def test_metrics_type_registration(self, metrics_aggregator):
        """测试指标类型注册"""
        # 注册自定义指标类型
        @metrics_aggregator.register_metric_type
        class CustomMetric:
            def __init__(self, value):
                self.value = value
                
            def validate(self):
                return self.value >= 0
                
        # 使用注册的类型
        metrics_aggregator.add_metrics('custom', CustomMetric(10))
        assert metrics_aggregator.get_latest('custom').value == 10
        
        # 测试无效类型
        with pytest.raises(ValueError):
            metrics_aggregator.add_metrics('custom', CustomMetric(-1))

    def test_metrics_storage_errors(self, metrics_aggregator):
        """测试存储错误处理"""
        # 测试存储路径不可写
        with patch('os.access', return_value=False):
            with pytest.raises(IOError):
                MetricsAggregator(storage_path="/root/forbidden")
                
        # 测试存储空间不足
        def raise_disk_full(*args):
            raise OSError("No space left on device")
            
        with patch.object(metrics_aggregator, '_write_to_disk', 
                         side_effect=raise_disk_full):
            # 应该继续工作，但记录错误
            metrics_aggregator.add_value('test', 1)
            assert metrics_aggregator.get_latest('test') == 1
            assert metrics_aggregator.storage_errors > 0

    def test_metrics_thread_safety_comprehensive(self, metrics_aggregator):
        """全面的线程安全测试"""
        import threading
        import queue
        
        results = queue.Queue()
        errors = queue.Queue()
        
        def worker(worker_id):
            try:
                with metrics_aggregator.transaction():
                    metrics_aggregator.add_value(f'thread_{worker_id}', worker_id)
                    # 模拟一些处理时间
                    time.sleep(random.random() * 0.01)
                    value = metrics_aggregator.get_latest(f'thread_{worker_id}')
                    results.put((worker_id, value))
            except Exception as e:
                errors.put((worker_id, e))
                
        # 创建多个线程并发操作
        threads = [threading.Thread(target=worker, args=(i,)) 
                  for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
            
        # 验证结果
        assert errors.empty()  # 不应有错误
        while not results.empty():
            worker_id, value = results.get()
            assert value == worker_id  # 每个线程应该看到自己的值

    @pytest.mark.benchmark
    def test_metrics_performance_monitoring(self, metrics_aggregator):
        """测试性能监控"""
        # 启用性能监控
        metrics_aggregator.enable_performance_monitoring()
        
        # 执行一些操作并测量性能
        start_time = time.time()
        for i in range(10000):
            metrics_aggregator.add_value('perf_test', i)
            
        # 获取性能指标
        perf_stats = metrics_aggregator.get_performance_stats()
        assert 'avg_operation_time' in perf_stats
        assert 'peak_memory_usage' in perf_stats
        assert 'operation_count' in perf_stats
        assert perf_stats['operation_count'] >= 10000
        
        # 验证监控对性能的影响
        total_time = time.time() - start_time
        assert total_time < 2.0  # 性能监控不应显著影响性能

    def test_metrics_lifecycle(self, metrics_aggregator):
        """测试指标生命周期"""
        # 创建
        metrics_aggregator.add_value('lifecycle', 1)
        assert metrics_aggregator.get_latest('lifecycle') == 1
        
        # 更新
        metrics_aggregator.update_value('lifecycle', 2)
        assert metrics_aggregator.get_latest('lifecycle') == 2
        
        # 过期
        with patch('time.time', return_value=time.time() + 7200):  # 2小时后
            assert metrics_aggregator.get_latest('lifecycle') is None
            
        # 删除
        metrics_aggregator.delete_metric('lifecycle')
        assert 'lifecycle' not in metrics_aggregator.get_metrics_names()

    def test_metrics_versioning(self, metrics_aggregator):
        """测试指标版本控制"""
        # 添加带版本的指标
        metrics_aggregator.add_value('versioned', 1, version='1.0')
        metrics_aggregator.add_value('versioned', 2, version='2.0')
        
        # 获取特定版本
        assert metrics_aggregator.get_latest('versioned', version='1.0') == 1
        assert metrics_aggregator.get_latest('versioned', version='2.0') == 2
        
        # 获取所有版本
        versions = metrics_aggregator.get_metric_versions('versioned')
        assert set(versions) == {'1.0', '2.0'}

    def test_metrics_config_validation_comprehensive(self):
        """全面的配置验证测试"""
        # 测试配置类型验证
        with pytest.raises(TypeError):
            MetricsAggregator(window_size='invalid')
            
        # 测试配置依赖关系
        with pytest.raises(ValueError):
            MetricsAggregator(
                max_points=100,
                batch_size=200  # batch_size不能大于max_points
            )
            
        # 测试可选配置默认值
        metrics = MetricsAggregator()
        assert metrics.storage_path is None  # 默认不使用持久化存储
        assert metrics.compression_enabled is False  # 默认不启用压缩

    def test_metrics_persistence_recovery(self, metrics_aggregator):
        """测试持久化和恢复"""
        # 添加测试数据
        metrics_aggregator.add_value('test', 1)
        metrics_aggregator.add_value('test', 2)
        
        # 保存到磁盘
        metrics_aggregator.persist()
        
        # 创建新的聚合器并加载数据
        new_aggregator = MetricsAggregator(
            storage_path="/tmp/metrics",
            window_size=3600
        )
        new_aggregator.load()
        
        # 验证数据已恢复
        assert new_aggregator.get_values('test') == [1, 2]

    @pytest.mark.asyncio
    async def test_metrics_recovery_strategies(self, metrics_aggregator):
        """测试恢复策略"""
        # 模拟存储失败后的内存模式
        with patch.object(metrics_aggregator, '_write_to_disk', 
                         side_effect=IOError):
            metrics_aggregator.add_value('test', 1)
            assert metrics_aggregator.is_memory_mode()
            
        # 模拟存储恢复
        metrics_aggregator._storage_available = True
        await metrics_aggregator.try_recover_storage()
        assert not metrics_aggregator.is_memory_mode()
        
        # 验证数据完整性
        assert metrics_aggregator.get_latest('test') == 1

    @pytest.mark.benchmark
    def test_metrics_performance_limits(self, metrics_aggregator):
        """测试性能限制"""
        metrics_aggregator.set_performance_limits(
            max_memory_mb=100,
            max_cpu_percent=50,
            max_disk_usage_mb=1000
        )
        
        # 添加大量数据直到触发限制
        try:
            for i in range(1000000):
                metrics_aggregator.add_value(f'test_{i}', i)
        except PerformanceError as e:
            assert "Resource limit exceeded" in str(e)
            
        # 验证资源使用在限制范围内
        stats = metrics_aggregator.get_resource_usage()
        assert stats['memory_mb'] <= 100
        assert stats['cpu_percent'] <= 50
        assert stats['disk_usage_mb'] <= 1000

    @pytest.mark.asyncio
    async def test_metrics_auto_recovery(self, metrics_aggregator):
        """测试自动恢复机制"""
        # 模拟连续失败
        failure_count = 0
        
        def failing_operation(*args):
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 3:  # 前3次失败
                raise RuntimeError("Operation failed")
            return True  # 第4次成功
            
        with patch.object(metrics_aggregator, '_write_to_disk',
                         side_effect=failing_operation):
            # 应该自动重试
            metrics_aggregator.add_value('test', 1)
            await asyncio.sleep(0.1)  # 等待重试
            assert metrics_aggregator.get_latest('test') == 1
            assert failure_count == 4  # 验证重试次数

    def test_metrics_cleanup_comprehensive(self, metrics_aggregator):
        """全面的清理测试"""
        # 添加不同类型的数据
        metrics_aggregator.add_value('numeric', 1)
        metrics_aggregator.add_metrics('connection', ConnectionMetrics(latency=0.1))
        metrics_aggregator.add_sample('timeseries', {
            'timestamp': time.time(),
            'value': 42
        })
        
        # 执行清理
        metrics_aggregator.cleanup(
            max_age=3600,
            max_points=100,
            metric_types=['numeric', 'connection']
        )
        
        # 验证清理结果
        assert metrics_aggregator.get_metrics_names() == {'timeseries'}
        assert not metrics_aggregator.has_metric('numeric')
        assert not metrics_aggregator.has_metric('connection')
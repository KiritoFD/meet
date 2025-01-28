import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from connect.jitsi.monitor import JitsiMonitor
from connect.jitsi.metrics import (
    ConnectionMetrics, DataMetrics,
    MeetingMetrics, SystemMetrics
)
import time
import sys
import os
import random
import math
from connect.jitsi.errors import RateLimitExceeded, ResourceQuotaExceeded  # 用于速率限制测试

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
async def monitor(config):
    monitor = JitsiMonitor(
        config=config,
        check_interval=1.0,
        metrics_window=60
    )
    try:
        yield monitor
    finally:
        try:
            # 确保清理所有资源
            await monitor.stop()
            monitor.clear_all_metrics()
            monitor.remove_all_collectors()
            monitor.remove_all_alert_rules()
            # 清理临时文件
            for file in [
                '/tmp/monitor_state.json',
                '/tmp/monitor',
                '/tmp/monitor_backup.zip'
            ]:
                if os.path.exists(file):
                    if os.path.isdir(file):
                        import shutil
                        shutil.rmtree(file)
                    else:
                        os.remove(file)
        except Exception as e:
            print(f"Warning: Failed to cleanup resources: {e}")
            # 不抛出异常，避免掩盖测试失败

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

    @pytest.mark.asyncio
    async def test_monitor_lifecycle(self, monitor):
        """测试监控器生命周期"""
        assert not monitor.is_running
        
        await monitor.start()
        assert monitor.is_running
        
        await monitor.stop()
        assert not monitor.is_running

    @pytest.mark.asyncio
    async def test_metrics_collection(self, monitor):
        """测试指标收集"""
        await monitor.start()
        
        # 添加测试指标
        monitor.add_metric('connection', ConnectionMetrics(
            latency=0.1,
            packet_loss=0.01,
            bandwidth=1000
        ))
        
        # 验证指标存储
        metrics = monitor.get_metrics('connection')
        assert len(metrics) == 1
        assert metrics[0].latency == 0.1

    @pytest.mark.asyncio
    async def test_alert_rules(self, monitor):
        """测试告警规则"""
        # 配置告警规则
        monitor.add_alert_rule(
            'high_latency',
            lambda m: m.get_latest('connection').latency > 0.5,
            cooldown=5
        )
        
        # 触发告警
        monitor.add_metric('connection', ConnectionMetrics(latency=0.6))
        alerts = monitor.get_active_alerts()
        assert len(alerts) == 1
        assert alerts[0].rule_id == 'high_latency'

    @pytest.mark.asyncio
    async def test_monitor_aggregation(self, monitor):
        """测试监控数据聚合"""
        await monitor.start()
        
        # 添加多个指标
        for i in range(5):
            monitor.add_metric('system', SystemMetrics(
                cpu_usage=0.2 * i,
                memory_usage=0.1 * i
            ))
            
        # 测试聚合计算
        stats = monitor.get_stats('system')
        assert stats['avg_cpu_usage'] == 0.4  # (0 + 0.2 + 0.4 + 0.6 + 0.8) / 5
        assert stats['max_memory_usage'] == 0.4  # max(0, 0.1, 0.2, 0.3, 0.4)

    @pytest.mark.asyncio
    async def test_monitor_thresholds(self, monitor):
        """测试阈值监控"""
        # 设置阈值
        monitor.set_threshold('cpu_usage', warning=0.7, critical=0.9)
        monitor.set_threshold('memory_usage', warning=0.8, critical=0.95)
        
        # 测试不同级别
        monitor.add_metric('system', SystemMetrics(cpu_usage=0.8))
        assert monitor.get_status('cpu_usage') == 'warning'
        
        monitor.add_metric('system', SystemMetrics(cpu_usage=0.95))
        assert monitor.get_status('cpu_usage') == 'critical'

    @pytest.mark.asyncio
    async def test_monitor_error_handling(self, monitor):
        """测试错误处理"""
        # 模拟指标收集失败
        async def failing_collector():
            raise RuntimeError("Collection failed")
            
        monitor.add_collector('failing', failing_collector)
        await monitor.start()
        
        # 应该继续运行并记录错误
        await asyncio.sleep(1.1)  # 等待一个检查周期
        assert monitor.is_running
        assert monitor.collection_errors > 0

    @pytest.mark.asyncio
    async def test_monitor_recovery(self, monitor):
        """测试恢复机制"""
        failure_count = 0
        
        async def unstable_collector():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:
                raise RuntimeError("Temporary failure")
            return SystemMetrics(cpu_usage=0.5)
            
        monitor.add_collector('unstable', unstable_collector)
        await monitor.start()
        
        # 等待恢复
        await asyncio.sleep(3.0)
        assert monitor.get_collector_status('unstable') == 'healthy'

    @pytest.mark.asyncio
    async def test_monitor_performance(self, monitor):
        """测试监控性能"""
        # 启用性能跟踪
        monitor.enable_performance_tracking()
        await monitor.start()
        
        # 添加大量指标
        for i in range(1000):
            monitor.add_metric('test', SystemMetrics(cpu_usage=0.1))
            
        # 验证性能指标
        perf_stats = monitor.get_performance_stats()
        assert perf_stats['avg_collection_time'] < 0.1
        assert perf_stats['memory_usage_mb'] < 50

    @pytest.mark.asyncio
    async def test_monitor_cleanup(self, monitor):
        """测试数据清理"""
        await monitor.start()
        
        # 添加过期数据
        with patch('time.time', return_value=time.time() - 3600):
            monitor.add_metric('old', SystemMetrics(cpu_usage=0.1))
            
        # 添加新数据
        monitor.add_metric('new', SystemMetrics(cpu_usage=0.2))
        
        # 触发清理
        monitor.cleanup(max_age=1800)  # 30分钟
        
        assert not monitor.has_metrics('old')
        assert monitor.has_metrics('new')

    @pytest.mark.asyncio
    async def test_monitor_data_retention(self, monitor):
        """测试数据保留策略"""
        await monitor.start()
        
        # 添加数据
        for i in range(200):
            monitor.add_metric('test', SystemMetrics(
                cpu_usage=0.1,
                memory_usage=0.2,
                timestamp=time.time() - i * 60  # 每分钟一个点
            ))
            
        # 验证数据保留
        assert monitor.get_metrics_count('test') <= monitor.max_points
        oldest_metric = monitor.get_oldest_metric('test')
        assert time.time() - oldest_metric.timestamp <= monitor.metrics_window

    @pytest.mark.asyncio
    async def test_monitor_alert_correlation(self, monitor):
        """测试告警关联分析"""
        # 配置多个告警规则
        monitor.add_alert_rule(
            'high_cpu',
            lambda m: m.get_latest('system').cpu_usage > 0.8
        )
        monitor.add_alert_rule(
            'high_memory',
            lambda m: m.get_latest('system').memory_usage > 0.8
        )
        
        # 触发相关告警
        monitor.add_metric('system', SystemMetrics(
            cpu_usage=0.9,
            memory_usage=0.9
        ))
        
        # 验证告警关联
        correlations = monitor.get_alert_correlations()
        assert len(correlations) > 0
        assert correlations[0].related_rules == {'high_cpu', 'high_memory'}

    @pytest.mark.asyncio
    async def test_monitor_error_handling_comprehensive(self, monitor):
        """全面的错误处理测试"""
        # 测试配置错误
        with pytest.raises(ValueError):
            monitor.set_threshold('cpu_usage', warning=0.9, critical=0.8)  # 警告阈值大于严重阈值
            
        # 测试无效的指标类型
        with pytest.raises(ValueError):
            monitor.add_metric('invalid_type', object())
            
        # 测试并发错误
        async def concurrent_operation():
            monitor.start_collection()
            await asyncio.sleep(0.1)
            monitor.stop_collection()
            
        with pytest.raises(RuntimeError):
            await asyncio.gather(
                concurrent_operation(),
                concurrent_operation()
            )

    @pytest.mark.asyncio
    async def test_monitor_collector_management(self, monitor):
        """测试采集器管理"""
        # 添加采集器
        async def custom_collector():
            return SystemMetrics(cpu_usage=0.5)
            
        monitor.add_collector('custom', custom_collector)
        assert 'custom' in monitor.get_collectors()
        
        # 暂停采集器
        monitor.pause_collector('custom')
        assert monitor.get_collector_status('custom') == 'paused'
        
        # 恢复采集器
        monitor.resume_collector('custom')
        assert monitor.get_collector_status('custom') == 'running'
        
        # 删除采集器
        monitor.remove_collector('custom')
        assert 'custom' not in monitor.get_collectors()

    @pytest.mark.asyncio
    async def test_monitor_state_transitions(self, monitor):
        """测试监控器状态转换"""
        # 初始状态
        assert monitor.state == 'initialized'
        
        # 启动
        await monitor.start()
        assert monitor.state == 'running'
        
        # 暂停
        await monitor.pause()
        assert monitor.state == 'paused'
        
        # 恢复
        await monitor.resume()
        assert monitor.state == 'running'
        
        # 停止
        await monitor.stop()
        assert monitor.state == 'stopped'
        
        # 验证状态转换约束
        with pytest.raises(RuntimeError):
            await monitor.resume()  # 不能从stopped状态恢复

    @pytest.mark.asyncio
    async def test_monitor_data_persistence(self, monitor):
        """测试监控数据持久化"""
        await monitor.start()
        
        # 添加测试数据
        monitor.add_metric('test', SystemMetrics(cpu_usage=0.5))
        
        # 保存状态
        state_file = '/tmp/monitor_state.json'
        await monitor.save_state(state_file)
        
        # 创建新的监控器并恢复状态
        new_monitor = JitsiMonitor(config=monitor.config)
        await new_monitor.load_state(state_file)
        
        # 验证数据恢复
        restored_metrics = new_monitor.get_metrics('test')
        assert len(restored_metrics) == 1
        assert restored_metrics[0].cpu_usage == 0.5

    @pytest.mark.asyncio
    async def test_monitor_error_recovery(self, monitor):
        """测试错误恢复机制"""
        error_count = 0
        recovery_count = 0
        
        @monitor.on_error
        async def error_handler(error):
            nonlocal error_count
            error_count += 1
            
        @monitor.on_recovery
        async def recovery_handler():
            nonlocal recovery_count
            recovery_count += 1
            
        # 模拟连续失败和恢复
        for i in range(3):
            await monitor._handle_collection_error(RuntimeError("Test error"))
            await asyncio.sleep(0.1)
            await monitor._try_recovery()
            
        assert error_count == 3
        assert recovery_count >= 1  # 至少有一次成功恢复

    @pytest.mark.asyncio
    async def test_monitor_resource_limits(self, monitor):
        """测试资源限制"""
        # 设置资源限制
        monitor.set_resource_limits(
            max_metrics=1000,
            max_memory_mb=50,
            max_collectors=5
        )
        
        # 测试指标数量限制
        for i in range(1500):
            monitor.add_metric(f'test_{i}', SystemMetrics(cpu_usage=0.1))
        assert monitor.get_total_metrics_count() <= 1000
        
        # 测试采集器数量限制
        for i in range(10):
            with pytest.raises(RuntimeError):
                monitor.add_collector(f'collector_{i}', lambda: None)

    @pytest.mark.asyncio
    async def test_monitor_event_handling(self, monitor):
        """测试事件处理"""
        events = []
        
        @monitor.on('metric_added')
        def handle_metric(metric_type, metric):
            events.append(('added', metric_type))
            
        @monitor.on('threshold_exceeded')
        def handle_threshold(metric_type, value, threshold):
            events.append(('exceeded', metric_type))
            
        @monitor.on('collector_failed')
        def handle_failure(collector_id, error):
            events.append(('failed', collector_id))
            
        # 触发事件
        monitor.add_metric('test', SystemMetrics(cpu_usage=0.95))  # 应触发 metric_added
        monitor.set_threshold('cpu_usage', critical=0.9)  # 应触发 threshold_exceeded
        
        assert ('added', 'test') in events
        assert ('exceeded', 'cpu_usage') in events

    @pytest.mark.asyncio
    async def test_monitor_batch_operations(self, monitor):
        """测试批量操作"""
        # 批量添加指标
        metrics = [
            SystemMetrics(cpu_usage=0.1 * i) for i in range(10)
        ]
        
        async with monitor.batch_context():
            for metric in metrics:
                monitor.add_metric('system', metric)
                
        # 验证批量处理结果
        assert monitor.get_metrics_count('system') == 10
        
        # 批量更新阈值
        thresholds = {
            'cpu_usage': {'warning': 0.7, 'critical': 0.9},
            'memory_usage': {'warning': 0.8, 'critical': 0.95}
        }
        
        monitor.set_thresholds(thresholds)
        assert all(monitor.get_threshold(k) == v 
                  for k, v in thresholds.items())

    def test_monitor_config_validation(self):
        """测试配置验证"""
        # 测试必需参数
        with pytest.raises(ValueError):
            JitsiMonitor({})  # 缺少必需配置
            
        # 测试参数类型
        with pytest.raises(TypeError):
            JitsiMonitor({
                'monitor': {
                    'window_size': 'invalid',  # 应该是数字
                    'sample_interval': 1.0
                }
            })
            
        # 测试参数范围
        with pytest.raises(ValueError):
            JitsiMonitor({
                'monitor': {
                    'window_size': -1,  # 不能为负
                    'sample_interval': 1.0
                }
            })
            
        # 测试参数依赖关系
        with pytest.raises(ValueError):
            JitsiMonitor({
                'monitor': {
                    'window_size': 10,
                    'sample_interval': 20  # 不能大于窗口大小
                }
            }) 

    @pytest.mark.asyncio
    async def test_monitor_concurrency(self, monitor):
        """测试并发安全性"""
        await monitor.start()
        
        # 并发添加指标
        async def add_metrics():
            for i in range(100):
                monitor.add_metric(
                    'concurrent',
                    SystemMetrics(cpu_usage=0.1 * i)
                )
                await asyncio.sleep(0.001)
                
        # 并发读取指标
        async def read_metrics():
            for _ in range(100):
                monitor.get_metrics('concurrent')
                await asyncio.sleep(0.001)
                
        # 同时执行读写操作
        await asyncio.gather(
            add_metrics(),
            add_metrics(),
            read_metrics(),
            read_metrics()
        )
        
        # 验证数据一致性
        metrics = monitor.get_metrics('concurrent')
        assert len(set(m.cpu_usage for m in metrics)) == 100  # 所有值都应该不同 

    @pytest.mark.asyncio
    async def test_monitor_metrics_filtering(self, monitor):
        """测试指标过滤"""
        await monitor.start()
        
        # 添加测试数据
        for i in range(10):
            monitor.add_metric('system', SystemMetrics(
                cpu_usage=0.1 * i,
                memory_usage=0.2 * i,
                thread_count=i
            ))
            
        # 测试条件过滤
        high_cpu = monitor.filter_metrics('system', 
            lambda m: m.cpu_usage > 0.5)
        assert len(high_cpu) == 4  # 0.6, 0.7, 0.8, 0.9
        
        # 测试时间范围过滤
        recent = monitor.filter_metrics('system',
            time_range=('1m', 'now'))  # 最近1分钟
        assert len(recent) <= 10

    @pytest.mark.asyncio
    async def test_monitor_alert_suppression(self, monitor):
        """测试告警抑制"""
        # 配置告警规则
        monitor.add_alert_rule('high_cpu', 
            lambda m: m.get_latest('system').cpu_usage > 0.8,
            suppression_interval=60  # 60秒内抑制重复告警
        )
        
        # 连续触发告警
        for i in range(5):
            monitor.add_metric('system', SystemMetrics(cpu_usage=0.9))
            await asyncio.sleep(0.1)
            
        # 验证告警抑制
        alerts = monitor.get_alerts_in_window(60)
        assert len(alerts) == 1  # 应该只有一个告警被触发 

    @pytest.mark.asyncio
    async def test_monitor_edge_cases(self, monitor):
        """测试边界情况"""
        await monitor.start()
        
        # 测试空数据处理
        assert monitor.get_metrics('nonexistent') == []
        assert monitor.get_latest('nonexistent') is None
        
        # 测试无效操作
        with pytest.raises(RuntimeError):
            await monitor.stop()  # 重复停止
            await monitor.stop()
            
        with pytest.raises(ValueError):
            monitor.set_threshold('cpu_usage', warning=1.5)  # 超出有效范围
            
        # 测试极限值
        monitor.add_metric('system', SystemMetrics(
            cpu_usage=1.0,
            memory_usage=0.0,
            thread_count=sys.maxsize
        ))
        assert monitor.get_latest('system') is not None 

    @pytest.mark.asyncio
    async def test_monitor_custom_metrics(self, monitor):
        """测试自定义指标"""
        class CustomMetric:
            def __init__(self, value):
                self.value = value
                self.timestamp = time.time()
                
            def validate(self):
                return isinstance(self.value, (int, float))
                
        # 注册自定义指标类型
        monitor.register_metric_type('custom', CustomMetric)
        
        # 添加自定义指标
        monitor.add_metric('custom', CustomMetric(42))
        
        # 验证指标处理
        latest = monitor.get_latest('custom')
        assert isinstance(latest, CustomMetric)
        assert latest.value == 42

    @pytest.mark.asyncio
    async def test_monitor_dynamic_thresholds(self, monitor):
        """测试动态阈值"""
        await monitor.start()
        
        # 添加历史数据
        for i in range(100):
            monitor.add_metric('system', SystemMetrics(
                cpu_usage=0.5 + random.random() * 0.1  # 正常范围: 0.5-0.6
            ))
            
        # 配置动态阈值
        monitor.set_dynamic_threshold(
            'cpu_usage',
            window_size=50,
            deviation_factor=2.0  # 超过2个标准差视为异常
        )
        
        # 测试正常值
        monitor.add_metric('system', SystemMetrics(cpu_usage=0.55))
        assert not monitor.is_anomaly('system', 'cpu_usage')
        
        # 测试异常值
        monitor.add_metric('system', SystemMetrics(cpu_usage=0.8))
        assert monitor.is_anomaly('system', 'cpu_usage') 

    def test_monitor_config_comprehensive(self):
        """全面的配置测试"""
        config = {
            'monitor': {
                'window_size': 3600,
                'sample_interval': 1.0,
                'storage': {
                    'type': 'file',
                    'path': '/tmp/monitor',
                    'retention': {
                        'max_age_days': 30,
                        'max_size_gb': 1
                    }
                },
                'alert': {
                    'channels': ['email', 'slack'],
                    'templates_path': '/etc/monitor/templates',
                    'rules_path': '/etc/monitor/rules'
                },
                'collectors': {
                    'system': {'enabled': True, 'interval': 60},
                    'network': {'enabled': False}
                }
            }
        }
        
        monitor = JitsiMonitor(config)
        
        # 验证配置加载
        assert monitor.window_size == 3600
        assert monitor.storage_config['type'] == 'file'
        assert monitor.alert_channels == ['email', 'slack']
        assert monitor.collector_configs['system']['enabled']
        assert not monitor.collector_configs['network']['enabled']

    @pytest.mark.asyncio
    async def test_monitor_cleanup_comprehensive(self, monitor):
        """全面的清理测试"""
        await monitor.start()
        
        # 添加各种类型的数据
        monitor.add_metric('system', SystemMetrics(cpu_usage=0.5))
        monitor.add_collector('test', lambda: None)
        monitor.add_alert_rule('test', lambda x: True)
        
        # 测试选择性清理
        await monitor.cleanup(
            metric_types=['system'],
            older_than='1h',
            include_collectors=True,
            include_rules=False
        )
        
        # 验证清理结果
        assert not monitor.has_metrics('system')
        assert not monitor.has_collector('test')
        assert monitor.has_alert_rule('test') 

    @pytest.mark.asyncio
    async def test_monitor_metrics_aggregation_methods(self, monitor):
        """测试指标聚合方法"""
        await monitor.start()
        
        # 添加测试数据
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        for v in values:
            monitor.add_metric('system', SystemMetrics(cpu_usage=v))
            
        # 测试不同聚合方法
        assert monitor.aggregate('system', 'cpu_usage', method='avg') == 0.3
        assert monitor.aggregate('system', 'cpu_usage', method='max') == 0.5
        assert monitor.aggregate('system', 'cpu_usage', method='min') == 0.1
        assert monitor.aggregate('system', 'cpu_usage', method='sum') == 1.5
        assert monitor.aggregate('system', 'cpu_usage', method='count') == 5

    @pytest.mark.asyncio
    async def test_monitor_metric_validation(self, monitor):
        """测试指标验证"""
        await monitor.start()
        
        # 测试无效的指标值
        with pytest.raises(ValueError):
            monitor.add_metric('system', SystemMetrics(cpu_usage=1.5))  # 超过1.0
            
        with pytest.raises(ValueError):
            monitor.add_metric('system', SystemMetrics(cpu_usage=-0.1))  # 小于0
            
        # 测试缺失必需字段
        with pytest.raises(ValueError):
            monitor.add_metric('system', {'memory_usage': 0.5})  # 缺少cpu_usage
            
        # 测试类型错误
        with pytest.raises(TypeError):
            monitor.add_metric('system', {'cpu_usage': '50%'})  # 应该是数字 

    @pytest.mark.asyncio
    async def test_monitor_async_context(self, monitor):
        """测试异步上下文管理器"""
        async with monitor.monitoring_session() as session:
            # 在会话中添加指标
            await session.add_metric('system', SystemMetrics(cpu_usage=0.5))
            
            # 验证会话中的指标
            metrics = await session.get_metrics('system')
            assert len(metrics) == 1
            assert metrics[0].cpu_usage == 0.5
            
        # 验证会话结束后指标仍然存在
        assert monitor.get_metrics_count('system') == 1
        
        # 验证会话已正确关闭 

    def test_monitor_initialization_params(self):
        """测试监控器初始化参数"""
        # 测试默认参数
        monitor = JitsiMonitor()
        assert monitor.check_interval == 1.0
        assert monitor.metrics_window == 3600
        assert monitor.max_points == 10000
        
        # 测试自定义参数
        monitor = JitsiMonitor(
            check_interval=2.0,
            metrics_window=7200,
            max_points=20000
        )
        assert monitor.check_interval == 2.0
        assert monitor.metrics_window == 7200
        assert monitor.max_points == 20000
        
        # 测试无效参数
        with pytest.raises(ValueError):
            JitsiMonitor(check_interval=0)  # 不能为0
        with pytest.raises(ValueError):
            JitsiMonitor(metrics_window=-1)  # 不能为负 

    @pytest.mark.asyncio
    async def test_monitor_state_recovery(self, monitor):
        """测试状态恢复"""
        await monitor.start()
        
        # 添加初始数据
        monitor.add_metric('system', SystemMetrics(cpu_usage=0.5))
        monitor.add_alert_rule('test', lambda x: x > 0.8)
        
        # 模拟崩溃
        await monitor.stop()
        state = monitor.save_state()
        
        # 创建新实例并恢复
        new_monitor = JitsiMonitor()
        new_monitor.restore_state(state)
        await new_monitor.start()
        
        # 验证状态恢复
        assert new_monitor.get_metrics_count('system') == 1
        assert new_monitor.has_alert_rule('test')
        assert new_monitor.is_running 

    @pytest.mark.asyncio
    async def test_monitor_performance_optimization(self, monitor):
        """测试性能优化"""
        await monitor.start()
        
        # 启用性能优化
        monitor.enable_optimizations(
            cache_enabled=True,
            batch_processing=True,
            compression_enabled=True
        )
        
        # 测试大批量数据处理
        start_time = time.time()
        metrics = [
            SystemMetrics(cpu_usage=0.1 * (i % 10))
            for i in range(10000)
        ]
        
        # 批量添加
        await monitor.add_metrics_batch('system', metrics)
        
        # 验证处理时间
        processing_time = time.time() - start_time
        assert processing_time < 1.0  # 应该在1秒内完成
        
        # 验证内存使用
        memory_usage = monitor.get_memory_usage()
        assert memory_usage['resident_mb'] < 100  # 内存使用应该受控 

    @pytest.mark.asyncio
    async def test_monitor_event_subscription(self, monitor):
        """测试事件订阅"""
        events = []
        
        # 订阅多个事件
        @monitor.subscribe('metrics.added', 'metrics.updated', 'metrics.removed')
        async def handle_metric_events(event_type, data):
            events.append((event_type, data))
            
        # 触发事件
        await monitor.start()
        monitor.add_metric('test', SystemMetrics(cpu_usage=0.5))
        monitor.update_metric('test', SystemMetrics(cpu_usage=0.6))
        monitor.remove_metric('test')
        
        # 验证事件处理
        assert len(events) == 3
        assert events[0][0] == 'metrics.added'
        assert events[1][0] == 'metrics.updated'
        assert events[2][0] == 'metrics.removed' 

    @pytest.mark.asyncio
    async def test_monitor_statistics(self, monitor):
        """测试监控统计功能"""
        await monitor.start()
        
        # 添加测试数据
        for i in range(100):
            monitor.add_metric('system', SystemMetrics(
                cpu_usage=0.1 * (i % 10),
                memory_usage=0.2 * (i % 5),
                thread_count=i % 20
            ))
            
        # 测试统计计算
        stats = monitor.get_statistics('system')
        assert 'cpu_usage' in stats
        assert 'memory_usage' in stats
        assert 'thread_count' in stats
        
        # 验证统计值
        cpu_stats = stats['cpu_usage']
        assert 0 <= cpu_stats['mean'] <= 1
        assert 'std_dev' in cpu_stats
        assert 'percentiles' in cpu_stats
        assert len(cpu_stats['percentiles']) == 3  # p50, p90, p99 

    @pytest.mark.asyncio
    async def test_monitor_data_export(self, monitor):
        """测试数据导出功能"""
        await monitor.start()
        
        # 添加测试数据
        for i in range(10):
            monitor.add_metric('system', SystemMetrics(
                cpu_usage=0.1 * i,
                memory_usage=0.2 * i
            ))
            
        # 测试不同格式导出
        json_data = monitor.export_data('system', format='json')
        assert isinstance(json_data, str)
        assert 'cpu_usage' in json_data
        
        csv_data = monitor.export_data('system', format='csv')
        assert isinstance(csv_data, str)
        assert 'timestamp,cpu_usage,memory_usage' in csv_data
        
        # 测试时间范围导出
        filtered_data = monitor.export_data(
            'system',
            time_range=('1h', 'now'),
            format='json'
        )
        assert isinstance(filtered_data, str) 

    @pytest.mark.asyncio
    async def test_monitor_metric_labels(self, monitor):
        """测试指标标签功能"""
        await monitor.start()
        
        # 添加带标签的指标
        monitor.add_metric('system', SystemMetrics(cpu_usage=0.5),
            labels={
                'host': 'server1',
                'environment': 'production',
                'region': 'us-east'
            })
            
        # 按标签查询
        metrics = monitor.get_metrics_by_labels('system', {
            'environment': 'production'
        })
        assert len(metrics) == 1
        
        # 按标签聚合
        aggregated = monitor.aggregate_by_labels('system', 
            group_by=['host', 'region'],
            method='avg'
        )
        assert 'server1' in aggregated
        assert 'us-east' in aggregated['server1'] 

    @pytest.mark.asyncio
    async def test_monitor_alert_templates(self, monitor):
        """测试告警模板功能"""
        # 注册告警模板
        monitor.register_alert_template('high_usage', """
            Resource: {resource}
            Current Usage: {value}%
            Threshold: {threshold}%
            Time: {timestamp}
        """)
        
        # 配置使用模板的告警规则
        monitor.add_alert_rule(
            'high_cpu',
            lambda m: m.get_latest('system').cpu_usage > 0.8,
            template='high_usage',
            template_vars={'resource': 'CPU'}
        )
        
        # 触发告警
        monitor.add_metric('system', SystemMetrics(cpu_usage=0.9))
        
        # 验证告警消息
        alerts = monitor.get_active_alerts()
        assert len(alerts) == 1
        assert 'Resource: CPU' in alerts[0].message
        assert '90%' in alerts[0].message 

    @pytest.mark.asyncio
    async def test_monitor_historical_analysis(self, monitor):
        """测试历史数据分析"""
        await monitor.start()
        
        # 添加历史数据
        timestamps = [time.time() - i * 3600 for i in range(24)]  # 24小时数据
        for ts in timestamps:
            monitor.add_metric('system', SystemMetrics(
                cpu_usage=0.5 + 0.1 * math.sin(ts / 3600 * math.pi),  # 模拟周期性变化
                timestamp=ts
            ))
            
        # 测试趋势分析
        trend = monitor.analyze_trend('system', 'cpu_usage')
        assert 'slope' in trend
        assert 'r_squared' in trend
        
        # 测试周期性检测
        periodicity = monitor.detect_periodicity('system', 'cpu_usage')
        assert abs(periodicity['period'] - 12) < 1  # 应检测出12小时周期

    @pytest.mark.asyncio
    async def test_monitor_notification_channels(self, monitor):
        """测试通知渠道"""
        # 配置通知渠道
        notifications = []
        
        async def mock_notify(alert):
            notifications.append(alert)
            
        monitor.register_notification_channel('email', mock_notify)
        monitor.register_notification_channel('slack', mock_notify)
        
        # 触发需要通知的告警
        monitor.add_metric('system', SystemMetrics(cpu_usage=0.95))
        await asyncio.sleep(0.1)  # 等待异步通知
        
        # 验证通知发送
        assert len(notifications) == 2  # 应该发送到两个渠道 

    @pytest.mark.asyncio
    async def test_monitor_retry_mechanism(self, monitor):
        """测试重试机制"""
        failure_count = 0
        
        async def failing_operation():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:
                raise ConnectionError("Network error")
            return True
            
        # 测试自动重试
        result = await monitor.retry_with_backoff(
            failing_operation,
            max_retries=3,
            initial_delay=0.1
        )
        
        assert result is True
        assert failure_count == 3  # 应该重试两次后成功
        
        # 测试超过最大重试次数
        failure_count = 0
        with pytest.raises(ConnectionError):
            await monitor.retry_with_backoff(
                failing_operation,
                max_retries=2  # 不足以成功
            ) 

    @pytest.mark.asyncio
    async def test_monitor_rate_limiting(self, monitor):
        """测试速率限制"""
        await monitor.start()
        
        # 配置速率限制
        monitor.set_rate_limits({
            'metrics_per_second': 100,
            'alerts_per_minute': 10
        })
        
        # 测试指标速率限制
        start_time = time.time()
        for i in range(200):
            monitor.add_metric('system', SystemMetrics(cpu_usage=0.5))
        elapsed = time.time() - start_time
        assert elapsed >= 2.0  # 应该至少需要2秒
        
        # 测试告警速率限制
        alerts_sent = 0
        for i in range(20):
            try:
                monitor.send_alert('test_alert', f'Alert {i}')
                alerts_sent += 1
            except RateLimitExceeded:
                break
        assert alerts_sent <= 10  # 每分钟最多10个告警

    @pytest.mark.asyncio
    async def test_monitor_data_sampling(self, monitor):
        """测试数据采样"""
        await monitor.start()
        
        # 添加大量数据点
        for i in range(1000):
            monitor.add_metric('system', SystemMetrics(
                cpu_usage=0.1 * (i % 10),
                timestamp=time.time() + i
            ))
            
        # 测试不同采样方法
        sampled_avg = monitor.get_sampled_data('system', 
            method='average',
            points=10
        )
        assert len(sampled_avg) == 10
        
        sampled_max = monitor.get_sampled_data('system',
            method='max',
            points=10
        )
        assert len(sampled_max) == 10
        assert all(m['value'] >= a['value'] 
                  for m, a in zip(sampled_max, sampled_avg)) 

    @pytest.mark.asyncio
    async def test_monitor_health_check(self, monitor):
        """测试健康检查"""
        await monitor.start()
        
        # 配置健康检查规则
        monitor.add_health_check('metrics_collection',
            lambda: monitor.collection_errors == 0)
        monitor.add_health_check('storage_available',
            lambda: monitor.storage_available)
        
        # 正常状态检查
        health = await monitor.check_health()
        assert health['status'] == 'healthy'
        assert all(check['status'] == 'pass' 
                  for check in health['checks'])
        
        # 模拟故障
        monitor.collection_errors += 1
        health = await monitor.check_health()
        assert health['status'] == 'unhealthy'
        assert any(check['status'] == 'fail' 
                  for check in health['checks']) 

    @pytest.mark.asyncio
    async def test_monitor_performance_profiling(self, monitor):
        """测试性能分析"""
        await monitor.start()
        
        # 启用性能分析
        with monitor.profile_operation('batch_insert'):
            # 执行批量操作
            metrics = [
                SystemMetrics(cpu_usage=0.1)
                for _ in range(1000)
            ]
            await monitor.add_metrics_batch('system', metrics)
            
        # 获取性能分析结果
        profile = monitor.get_operation_profile('batch_insert')
        assert 'duration' in profile
        assert 'memory_delta' in profile
        assert 'cpu_time' in profile
        
        # 验证性能指标
        assert profile['duration'] < 1.0  # 应该在1秒内完成
        assert profile['memory_delta'] < 50 * 1024 * 1024  # 内存增长应小于50MB 

    @pytest.mark.asyncio
    async def test_monitor_data_retention_policy(self, monitor):
        """测试数据保留策略"""
        await monitor.start()
        
        # 配置保留策略
        monitor.set_retention_policy({
            'max_age_days': 7,
            'max_points_per_metric': 1000,
            'storage_limit_mb': 100
        })
        
        # 添加过期数据
        old_time = time.time() - 8 * 24 * 3600  # 8天前
        monitor.add_metric('old_data', SystemMetrics(
            cpu_usage=0.5,
            timestamp=old_time
        ))
        
        # 触发清理
        await monitor.enforce_retention_policy()
        
        # 验证清理结果
        assert not monitor.has_metrics('old_data')
        assert monitor.get_storage_usage_mb() <= 100

    @pytest.mark.asyncio
    async def test_monitor_backup_restore(self, monitor):
        """测试备份恢复"""
        await monitor.start()
        
        # 添加测试数据
        monitor.add_metric('system', SystemMetrics(cpu_usage=0.5))
        monitor.add_alert_rule('test', lambda x: x > 0.8)
        
        # 创建备份
        backup_file = '/tmp/monitor_backup.zip'
        await monitor.create_backup(backup_file)
        
        # 清空当前数据
        monitor.clear_all()
        assert not monitor.has_metrics('system')
        
        # 从备份恢复
        await monitor.restore_from_backup(backup_file)
        
        # 验证恢复
        assert monitor.has_metrics('system')
        assert monitor.has_alert_rule('test') 

    @pytest.mark.asyncio
    async def test_monitor_error_handling_edge_cases(self, monitor):
        """测试边缘错误情况"""
        await monitor.start()
        
        # 测试无效的指标格式
        with pytest.raises(ValueError):
            monitor.add_metric('system', {'invalid': 'format'})
            
        # 测试重复注册
        with pytest.raises(ValueError):
            monitor.register_metric_type('system', SystemMetrics)
            
        # 测试无效的时间范围
        with pytest.raises(ValueError):
            monitor.get_metrics('system', time_range=('invalid', 'now'))
            
        # 测试并发限制
        with pytest.raises(RuntimeError):
            async with monitor.transaction():
                async with monitor.transaction():  # 嵌套事务
                    pass  # 添加这一行完成代码块 

    @pytest.mark.asyncio
    async def test_monitor_logging(self, monitor):
        """测试日志记录"""
        # 配置日志捕获
        log_messages = []
        
        def log_handler(level, message):
            log_messages.append((level, message))
            
        monitor.set_log_handler(log_handler)
        await monitor.start()
        
        # 触发不同级别的日志
        monitor.log_info("Starting metrics collection")
        monitor.log_warning("High CPU usage detected")
        monitor.log_error("Failed to store metrics")
        
        # 验证日志记录
        assert len(log_messages) == 3
        assert log_messages[0][0] == 'INFO'
        assert log_messages[1][0] == 'WARNING'
        assert log_messages[2][0] == 'ERROR'
        assert 'metrics collection' in log_messages[0][1] 

    @pytest.mark.asyncio
    async def test_monitor_metric_filters(self, monitor):
        """测试指标过滤器"""
        await monitor.start()
        
        # 注册过滤器
        @monitor.metric_filter('system')
        def validate_cpu_usage(metric):
            return 0 <= metric.cpu_usage <= 1
            
        @monitor.metric_filter('system')
        def validate_memory_usage(metric):
            return 0 <= metric.memory_usage <= 1
            
        # 测试有效指标
        monitor.add_metric('system', SystemMetrics(
            cpu_usage=0.5,
            memory_usage=0.5
        ))
        assert monitor.get_metrics_count('system') == 1
        
        # 测试无效指标
        with pytest.raises(ValueError):
            monitor.add_metric('system', SystemMetrics(
                cpu_usage=1.5,  # 超出范围
                memory_usage=0.5
            )) 

    @pytest.mark.asyncio
    async def test_monitor_metric_transformations(self, monitor):
        """测试指标转换"""
        await monitor.start()
        
        # 注册转换器
        @monitor.metric_transformer('system')
        def normalize_metrics(metric):
            return SystemMetrics(
                cpu_usage=metric.cpu_usage / 100 if metric.cpu_usage > 1 else metric.cpu_usage,
                memory_usage=metric.memory_usage / 100 if metric.memory_usage > 1 else metric.memory_usage
            )
            
        # 测试指标转换
        monitor.add_metric('system', SystemMetrics(
            cpu_usage=75,  # 百分比形式
            memory_usage=80
        ))
        
        # 验证转换结果
        metric = monitor.get_latest('system')
        assert 0 <= metric.cpu_usage <= 1  # 已转换为小数
        assert 0 <= metric.memory_usage <= 1 

    def test_monitor_config_inheritance(self):
        """测试配置继承和覆盖"""
        base_config = {
            'monitor': {
                'window_size': 3600,
                'alert': {
                    'channels': ['email'],
                    'cooldown': 300
                }
            }
        }
        
        override_config = {
            'monitor': {
                'alert': {
                    'channels': ['slack'],
                    'templates': {'custom': 'template'}
                }
            }
        }
        
        # 测试配置继承
        monitor = JitsiMonitor(base_config)
        monitor.update_config(override_config)
        
        # 验证配置合并结果
        assert monitor.window_size == 3600  # 保持基础配置
        assert monitor.alert_channels == ['slack']  # 被覆盖
        assert monitor.alert_cooldown == 300  # 保持基础配置
        assert monitor.alert_templates['custom'] == 'template'  # 新增配置 

    @pytest.mark.asyncio
    async def test_monitor_metric_dependencies(self, monitor):
        """测试指标依赖关系"""
        await monitor.start()
        
        # 注册依赖指标
        @monitor.derived_metric('system_health')
        def calculate_health(metrics):
            cpu = metrics.get_latest('system').cpu_usage
            memory = metrics.get_latest('system').memory_usage
            return (1 - cpu) * (1 - memory)  # 简单的健康度计算
            
        # 添加基础指标
        monitor.add_metric('system', SystemMetrics(
            cpu_usage=0.5,
            memory_usage=0.5
        ))
        
        # 验证派生指标
        health = monitor.get_latest('system_health')
        assert health == 0.25  # (1-0.5) * (1-0.5)
        
        # 测试依赖更新
        monitor.add_metric('system', SystemMetrics(
            cpu_usage=0.8,
            memory_usage=0.8
        ))
        health = monitor.get_latest('system_health')
        assert health == 0.04  # (1-0.8) * (1-0.8) 

    @pytest.mark.asyncio
    async def test_monitor_state_machine(self, monitor):
        """测试状态机"""
        # 定义有效的状态转换
        valid_transitions = {
            'initialized': ['running'],
            'running': ['paused', 'stopped'],
            'paused': ['running', 'stopped'],
            'stopped': ['initialized']  # 只能通过重新初始化恢复
        }
        
        # 测试所有有效转换
        for from_state, to_states in valid_transitions.items():
            monitor._state = from_state  # 直接设置状态用于测试
            for to_state in to_states:
                await monitor.transition_to(to_state)
                assert monitor.state == to_state
                
        # 测试无效转换
        monitor._state = 'stopped'
        with pytest.raises(RuntimeError):
            await monitor.transition_to('running')  # 不能从stopped直接到running 

    @pytest.mark.asyncio
    async def test_monitor_resource_quotas(self, monitor):
        """测试资源配额"""
        await monitor.start()
        
        # 设置资源配额
        monitor.set_resource_quotas({
            'max_metrics_per_type': 1000,
            'max_storage_size_mb': 100,
            'max_memory_percent': 75,
            'max_cpu_percent': 50
        })
        
        # 测试指标数量限制
        for i in range(1500):
            if i < 1000:
                monitor.add_metric(f'test_{i}', SystemMetrics(cpu_usage=0.5))
            else:
                with pytest.raises(ResourceQuotaExceeded):
                    monitor.add_metric(f'test_{i}', SystemMetrics(cpu_usage=0.5))
                    
        # 验证资源使用
        usage = monitor.get_resource_usage()
        assert usage['metrics_count'] <= 1000
        assert usage['storage_size_mb'] <= 100
        assert usage['memory_percent'] <= 75
        assert usage['cpu_percent'] <= 50 

    @pytest.mark.asyncio
    async def test_monitor_aggregation_rules(self, monitor):
        """测试聚合规则"""
        await monitor.start()
        
        # 配置聚合规则
        monitor.add_aggregation_rule('system_load', {
            'metrics': ['cpu_usage', 'memory_usage', 'disk_usage'],
            'method': 'weighted_average',
            'weights': {'cpu_usage': 0.5, 'memory_usage': 0.3, 'disk_usage': 0.2}
        })
        
        # 添加测试数据
        monitor.add_metric('system', SystemMetrics(
            cpu_usage=0.8,
            memory_usage=0.6,
            disk_usage=0.4
        ))
        
        # 验证聚合结果
        load = monitor.get_aggregated_value('system_load')
        expected = 0.8 * 0.5 + 0.6 * 0.3 + 0.4 * 0.2
        assert abs(load - expected) < 0.001 

    @pytest.mark.asyncio
    async def test_monitor_data_compression(self, monitor):
        """测试数据压缩"""
        await monitor.start()
        
        # 启用压缩
        monitor.enable_compression(algorithm='gzip', level=6)
        
        # 添加大量重复数据
        for i in range(1000):
            monitor.add_metric('system', SystemMetrics(
                cpu_usage=0.5,
                memory_usage=0.5
            ))
            
        # 验证压缩效果
        storage_size = monitor.get_storage_size()
        compressed_size = monitor.get_compressed_size()
        compression_ratio = compressed_size / storage_size
        assert compression_ratio < 0.5  # 至少50%的压缩率 

    @pytest.mark.asyncio
    async def test_monitor_integration(self, monitor, client, meeting_manager):
        """测试监控器与其他组件的集成"""
        await monitor.start()
        
        # 监听会议事件
        events = []
        @monitor.on('meeting_event')
        def handle_meeting_event(event_type, data):
            events.append((event_type, data))
        
        # 创建会议并加入
        meeting = await meeting_manager.create_meeting('test_room')
        await client.join_meeting(meeting.id)
        
        # 验证监控指标
        metrics = monitor.get_latest('meeting')
        assert metrics.room_id == meeting.id
        assert metrics.participant_count == 1
        
        # 验证事件捕获
        assert ('meeting_created', meeting.id) in events
        assert ('participant_joined', client.user_id) in events 

    @pytest.mark.asyncio
    async def test_monitor_fault_tolerance(self, monitor, transport):
        """测试监控器容错性"""
        await monitor.start()
        
        # 模拟网络故障
        transport.simulate_network_failure()
        
        # 添加指标应该进入缓存
        monitor.add_metric('system', SystemMetrics(cpu_usage=0.5))
        assert monitor.get_cached_metrics_count() > 0
        
        # 恢复网络
        transport.restore_network()
        await asyncio.sleep(1.0)  # 等待重试
        
        # 验证数据已同步
        assert monitor.get_cached_metrics_count() == 0
        assert monitor.get_metrics_count('system') == 1 

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_monitor_performance_benchmark(self, monitor):
        """测试监控器性能基准"""
        await monitor.start()
        
        # 测试高频指标收集
        start_time = time.time()
        metrics_count = 10000
        
        async def add_metrics():
            for i in range(metrics_count):
                monitor.add_metric('benchmark', SystemMetrics(
                    cpu_usage=random.random(),
                    memory_usage=random.random()
                ))
                
        await add_metrics()
        elapsed = time.time() - start_time
        
        # 验证性能指标
        assert elapsed < 5.0  # 5秒内完成1万次指标收集
        assert monitor.get_metrics_count('benchmark') == metrics_count
        
        # 验证内存使用
        memory_usage = monitor.get_memory_usage()
        assert memory_usage['resident_mb'] < 100  # 内存使用应控制在100MB以内 

    @pytest.mark.asyncio
    async def test_alert_handling():
        """测试告警触发机制"""
        alerts = []
        monitor = JitsiMonitor({})
        monitor.add_alert_handler(lambda a: alerts.append(a))
        
        # 触发CPU告警
        monitor.record_system_metrics(SystemMetrics(cpu_usage=0.9, memory_usage=0.5))
        assert len(alerts) == 1
        assert "CPU" in alerts[0]
        
        # 触发网络告警
        monitor.record_connection_metrics(ConnectionMetrics(packet_loss=0.85))
        assert len(alerts) == 2
        assert "packet loss" in alerts[1]

    @pytest.mark.asyncio
    async def test_connection_health_check(monitor):
        """测试连接健康检查"""
        # 记录正常指标
        for _ in range(3):
            monitor.record_connection_metrics(ConnectionMetrics(
                latency=0.2, packet_loss=0.05, state="connected"
            ))
        
        # 记录异常指标
        monitor.record_connection_metrics(ConnectionMetrics(
            latency=1.5, packet_loss=0.3, state="connected"
        ))
        
        # 执行健康检查
        await monitor._check_connection_health() 
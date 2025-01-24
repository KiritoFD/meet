import pytest
import time
import asyncio
from monitoring.jitsi.monitor import JitsiMonitor
from monitoring.jitsi.dashboard import JitsiDashboard

class TestJitsiMonitor:
    @pytest.fixture
    def setup_monitor(self):
        """初始化监控器"""
        config = {
            'metrics_interval': 5,
            'alert_threshold': 0.8,
            'history_size': 100,
            'alert_channels': ['email', 'slack']
        }
        return JitsiMonitor(config)
        
    def test_record_metrics(self, setup_monitor):
        """测试指标记录"""
        setup_monitor.record_metric('latency', 0.1)
        setup_monitor.record_metric('fps', 30)
        
        metrics = setup_monitor.metrics
        assert 'latency' in metrics
        assert 'fps' in metrics
        assert len(metrics['latency']) == 1
        assert metrics['fps'][0] == 30
        
    def test_alert_triggering(self, setup_monitor):
        """测试告警触发"""
        # 模拟高延迟
        for _ in range(10):
            setup_monitor.record_metric('latency', 0.5)  # 500ms
            
        alerts = setup_monitor.check_alerts()
        assert len(alerts) > 0
        assert any(a.metric == 'latency' for a in alerts)
        
    @pytest.mark.asyncio
    async def test_long_term_monitoring(self, setup_monitor):
        """测试长期监控"""
        # 模拟5分钟的数据收集
        start_time = time.time()
        while time.time() - start_time < 300:
            setup_monitor.record_metric('memory', 100 + time.time() - start_time)
            await asyncio.sleep(1)
            
        # 验证数据保留策略
        assert len(setup_monitor.metrics['memory']) <= setup_monitor._config['history_size'] 

    def test_metric_aggregation(self, setup_monitor):
        # 测试指标聚合
        for i in range(100):
            setup_monitor.record_metric('cpu', 50 + i % 10)
            setup_monitor.record_metric('memory', 500 + i)
            
        stats = setup_monitor.get_aggregated_stats()
        assert 45 <= stats['cpu']['avg'] <= 55
        assert stats['memory']['trend'] > 0

    @pytest.mark.asyncio
    async def test_alert_throttling(self, setup_monitor):
        # 测试告警节流
        for _ in range(100):
            setup_monitor.record_metric('latency', 1.0)  # 严重延迟
            
        # 应该只产生有限的告警
        alerts = await setup_monitor.get_active_alerts()
        assert len(alerts) <= 5  # 告警应该被合并或抑制

    def test_dashboard_data(self, setup_monitor):
        # 测试仪表盘数据生成
        setup_monitor.record_metric('fps', 30)
        setup_monitor.record_metric('latency', 0.1)
        
        dashboard_data = setup_monitor.get_dashboard_data()
        assert 'performance' in dashboard_data
        assert 'network' in dashboard_data
        assert len(dashboard_data['charts']) > 0

    @pytest.mark.asyncio
    async def test_periodic_cleanup(self, setup_monitor):
        # 测试定期清理
        old_time = time.time() - 3600
        setup_monitor.metrics['old_metric'] = [(old_time, 100)]
        
        await setup_monitor._cleanup_old_data()
        assert 'old_metric' not in setup_monitor.metrics 
import pytest
import time
import psutil
import threading
from connect.performance_monitor import PerformanceMonitor

class TestPerformance:
    def test_high_load(self, unified_sender, performance_monitor):
        """测试高负载"""
        # 记录初始状态
        process = psutil.Process()
        initial_cpu = process.cpu_percent()
        initial_memory = process.memory_info().rss
        
        # 发送大量数据
        for _ in range(1000):
            asyncio.run(unified_sender.send(
                data_type='pose',
                data=generate_test_pose()
            ))
            
        # 检查资源使用
        stats = performance_monitor.get_stats()
        assert stats['cpu_usage'] - initial_cpu < 30
        memory_growth = (process.memory_info().rss - initial_memory) / 1024 / 1024
        assert memory_growth < 100  # MB

    def test_concurrent_users(self, unified_sender, performance_monitor):
        """测试并发用户"""
        thread_count = 10
        sends_per_thread = 100
        
        def send_data():
            for _ in range(sends_per_thread):
                asyncio.run(unified_sender.send(
                    data_type='pose',
                    data=generate_test_pose()
                ))
                
        threads = [
            threading.Thread(target=send_data)
            for _ in range(thread_count)
        ]
        
        start_time = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
            
        duration = time.time() - start_time
        total_sends = thread_count * sends_per_thread
        
        stats = performance_monitor.get_stats()
        assert stats['success_rate'] > 0.95  # 95%成功率
        assert duration < total_sends / 30  # 保持至少30fps 

class TestPerformanceMonitor:
    @pytest.fixture
    def setup_monitor(self):
        """初始化性能监控器"""
        monitor = PerformanceMonitor()
        yield monitor
        monitor.cleanup()

    def test_metrics_collection(self, setup_monitor):
        """测试指标收集"""
        # 更新一些指标
        for _ in range(100):
            setup_monitor.update(
                success=True,
                latency=0.01,
                data_size=1000
            )
            
        stats = setup_monitor.get_stats()
        
        assert 'fps' in stats
        assert 'latency' in stats
        assert 'success_rate' in stats
        assert 'memory_usage' in stats
        assert stats['success_rate'] == 1.0

    def test_alerts(self, setup_monitor):
        """测试告警机制"""
        alerts = []
        setup_monitor.set_alert_callback(lambda a: alerts.append(a))
        
        # 触发高延迟告警
        for _ in range(10):
            setup_monitor.update(
                success=True,
                latency=0.2,  # 200ms, 高于阈值
                data_size=1000
            )
            
        assert len(alerts) > 0
        assert any('latency' in a['message'] for a in alerts)

    def test_memory_leak_detection(self, setup_monitor):
        """测试内存泄漏检测"""
        initial_memory = setup_monitor.get_stats()['memory_usage']
        
        # 模拟内存增长
        large_data = ['x' * 1000000]  # 创建大对象
        for _ in range(10):
            setup_monitor.update(
                success=True,
                latency=0.01,
                data_size=len(str(large_data))
            )
            
        # 检查内存增长告警
        assert setup_monitor.has_memory_leak_warning() 
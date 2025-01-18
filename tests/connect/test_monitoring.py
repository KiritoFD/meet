import pytest
from connect.performance_monitor import PerformanceMonitor
from connect.socket_manager import SocketManager
from connect.pose_sender import PoseSender

class TestMonitoring:
    def test_performance_metrics(self):
        """测试性能指标监控"""
        monitor = PerformanceMonitor()
        socket = SocketManager()
        sender = PoseSender(socket)
        
        monitor.attach(sender)
        
        # 发送100帧
        for _ in range(100):
            sender.send_pose_data(
                "test_room",
                self._generate_test_pose()
            )
        
        metrics = monitor.get_metrics()
        assert 'fps' in metrics
        assert 'latency' in metrics
        assert 'success_rate' in metrics
        assert 'memory_usage' in metrics
        
        # 验证指标范围
        assert 20 <= metrics['fps'] <= 40
        assert 0 <= metrics['latency'] <= 100
        assert 0.9 <= metrics['success_rate'] <= 1.0 
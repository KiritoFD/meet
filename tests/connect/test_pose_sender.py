import sys
from pathlib import Path
import pytest
from unittest.mock import Mock, patch
import numpy as np
import time
import threading
import queue
import gc
import psutil
from typing import List, Dict, Any

from connect.pose_sender import PoseSender
from connect.socket_manager import SocketManager
from connect.errors import (
    ConnectionError, 
    DataValidationError,
    ResourceLimitError,
    SecurityError
)

class TestPoseSender:
    """PoseSender组件的测试类
    
    测试范围:
    1. 基本功能测试
    2. 错误处理测试
    3. 性能测试
    4. 并发测试
    5. 安全性测试
    """

    @pytest.fixture
    def mock_socket_manager(self):
        """创建模拟的SocketManager
        
        Returns:
            Mock: 配置好的SocketManager模拟对象
        """
        mock_manager = Mock(spec=SocketManager)
        mock_manager.connected = True
        mock_manager.emit = Mock(return_value=True)
        return mock_manager

    @pytest.fixture
    def setup_sender(self, mock_socket_manager):
        """初始化测试环境
        
        Args:
            mock_socket_manager: 模拟的socket管理器
            
        Returns:
            PoseSender: 配置好的发送器实例
        """
        sender = PoseSender(mock_socket_manager)
        try:
            yield sender
        finally:
            # 确保资源清理
            sender.cleanup()

    @pytest.mark.timeout(5)
    def test_send_performance(self, setup_sender):
        """测试发送性能
        
        验证:
        1. 单帧处理时间 < 5ms
        2. 内存增长 < 1MB
        3. CPU使用率 < 50%
        4. 帧率稳定性
        """
        try:
            setup_sender.start_monitoring()
            
            frame_count = 100
            frame_times = []
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            initial_cpu = process.cpu_percent()

            for _ in range(frame_count):
                start_time = time.time()
                success = setup_sender.send_pose_data(
                    room="test_room",
                    pose_results=self._generate_test_pose(),
                    timestamp=time.time()
                )
                assert success, "发送失败"
                frame_times.append(time.time() - start_time)

            # 性能指标验证
            avg_time = sum(frame_times) / len(frame_times)
            memory_increase = (process.memory_info().rss - initial_memory) / 1024 / 1024
            cpu_usage = process.cpu_percent() - initial_cpu
            
            # 计算帧率稳定性
            frame_intervals = [frame_times[i+1] - frame_times[i] for i in range(len(frame_times)-1)]
            interval_std = np.std(frame_intervals)

            assert avg_time < 0.005, f"处理时间过长: {avg_time*1000:.2f}ms > 5ms"
            assert memory_increase < 1, f"内存增长过大: {memory_increase:.1f}MB > 1MB"
            assert cpu_usage < 50, f"CPU使用率过高: {cpu_usage:.1f}% > 50%"
            assert interval_std < 0.002, f"帧率不稳定: 标准差 {interval_std*1000:.2f}ms > 2ms"
            
        finally:
            setup_sender.stop_monitoring()

    def test_invalid_data_handling(self, setup_sender):
        """测试无效数据处理
        
        测试场景:
        1. None值
        2. 空数据
        3. 格式错误
        4. 数值越界
        5. 类型错误
        """
        invalid_cases = [
            (None, "空数据"),
            ({}, "空字典"),
            ({'landmarks': None}, "空landmarks"),
            ({'landmarks': []}, "空关键点列表"),
            ({'landmarks': [{'x': 'invalid'}]}, "无效坐标类型"),
            ({'landmarks': [{'x': float('inf')}]}, "无限大坐标"),
            ({'landmarks': [{'x': 0, 'y': None}]}, "缺失坐标"),
            ({'invalid_key': 'value'}, "未知字段"),
            ({'landmarks': [{'x': 0, 'y': 0, 'z': 2}]}, "坐标超范围"),
        ]

        for invalid_data, error_msg in invalid_cases:
            success = setup_sender.send_pose_data(
                room="test_room",
                pose_results=invalid_data,
                timestamp=time.time()
            )
            assert not success, f"应该拒绝无效数据 - {error_msg}: {invalid_data}"

    @pytest.mark.timeout(10)
    def test_thread_safety(self, setup_sender):
        """测试线程安全性
        
        验证:
        1. 多线程并发发送
        2. 资源竞争处理
        3. 错误传播
        """
        lock = threading.Lock()
        errors: List[Exception] = []
        success_count = 0
        total_count = 1000
        thread_count = 10
        
        def worker():
            nonlocal success_count
            try:
                for _ in range(total_count // thread_count):
                    if setup_sender.send_pose_data(
                        room="test_room",
                        pose_results=self._generate_test_pose(),
                        timestamp=time.time()
                    ):
                        with lock:
                            success_count += 1
            except Exception as e:
                with lock:
                    errors.append(e)

        # 启动多线程测试
        threads = [threading.Thread(target=worker) for _ in range(thread_count)]
        start_time = time.time()
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        elapsed = time.time() - start_time
        success_rate = success_count / total_count

        # 验证结果
        assert not errors, f"线程安全性测试失败: {errors}"
        assert success_rate >= 0.95, f"成功率过低: {success_rate*100:.1f}% < 95%"
        assert elapsed < 10, f"总执行时间过长: {elapsed:.1f}s > 10s"

    def test_send_pose_data(self, setup_sender, mock_socket_manager):
        """测试发送姿态数据"""
        # 创建测试数据
        pose_data = self._generate_test_pose()
        timestamp = time.time()

        # 测试发送
        success = setup_sender.send_pose_data(
            room="test_room",
            pose_results=pose_data,
            timestamp=timestamp
        )

        assert success
        mock_socket_manager.emit.assert_called_once()

    def test_send_performance(self, setup_sender):
        """测试发送性能"""
        frame_count = 100
        frame_times = []

        for _ in range(frame_count):
            start_time = time.time()
            setup_sender.send_pose_data(
                room="test_room",
                pose_results=self._generate_test_pose(),
                timestamp=time.time()
            )
            frame_times.append(time.time() - start_time)

        avg_time = sum(frame_times) / len(frame_times)
        assert avg_time < 0.005  # 每帧处理时间应小于5ms

    def test_error_handling(self, setup_sender, mock_socket_manager):
        """测试错误处理"""
        # 模拟发送失败
        mock_socket_manager.emit.side_effect = Exception("Send failed")

        success = setup_sender.send_pose_data(
            room="test_room",
            pose_results=self._generate_test_pose(),
            timestamp=time.time()
        )

        assert not success

    def test_queue_management(self, setup_sender):
        """测试队列管理"""
        # 设置较小的队列大小
        setup_sender.queue_size = 5
        
        # 快速发送多个帧
        for i in range(10):
            success = setup_sender.send_pose_data(
                room="test_room",
                pose_results=self._generate_test_pose(),
                timestamp=time.time()
            )
            if i >= 5:
                assert not success  # 队列已满

    def test_send_strategy(self, setup_sender):
        """测试发送策略"""
        # 测试帧率控制
        setup_sender.set_target_fps(30)
        frame_times = []
        
        for _ in range(100):
            start = time.time()
            setup_sender.send_pose_data(
                room="test_room",
                pose_results=self._generate_test_pose()
            )
            frame_times.append(time.time() - start)
        
        # 验证帧率
        avg_interval = sum(frame_times) / len(frame_times)
        assert abs(avg_interval - 1/30) < 0.005  # 允许5ms误差

    def test_compression_strategy(self, setup_sender):
        """测试压缩策略"""
        # 生成重复性高的数据
        repetitive_pose = self._generate_test_pose()
        
        # 测试自动压缩
        setup_sender.enable_compression(True)
        compressed_size = setup_sender._get_data_size(repetitive_pose)
        
        setup_sender.enable_compression(False)
        uncompressed_size = setup_sender._get_data_size(repetitive_pose)
        
        assert compressed_size < uncompressed_size

    def test_connection_status(self, setup_sender, mock_socket_manager):
        """测试连接状态检查"""
        # 测试已连接状态
        mock_socket_manager.connected = True
        assert setup_sender.is_connected()

        # 测试未连接状态
        mock_socket_manager.connected = False
        assert not setup_sender.is_connected()

    def test_room_management(self, setup_sender, mock_socket_manager):
        """测试房间管理功能"""
        # 测试加入房间
        setup_sender.join_room("test_room")
        mock_socket_manager.emit.assert_called_with('join', {'room': 'test_room'})

        # 测试离开房间
        setup_sender.leave_room("test_room")
        mock_socket_manager.emit.assert_called_with('leave', {'room': 'test_room'})

    def test_pose_data_format(self, setup_sender, mock_socket_manager):
        """测试姿态数据格式化"""
        pose_data = self._generate_test_pose()
        timestamp = time.time()

        setup_sender.send_pose_data(
            room="test_room",
            pose_results=pose_data,
            timestamp=timestamp
        )

        # 验证发送的数据格式
        called_args = mock_socket_manager.emit.call_args[0]
        assert called_args[0] == 'pose_data'  # 事件名称
        sent_data = called_args[1]
        
        assert 'room' in sent_data
        assert 'pose_results' in sent_data
        assert 'timestamp' in sent_data
        assert isinstance(sent_data['pose_results']['landmarks'], list)

    def test_invalid_data_handling(self, setup_sender):
        """测试无效数据处理"""
        # 测试空数据
        success = setup_sender.send_pose_data(
            room="test_room",
            pose_results=None,
            timestamp=time.time()
        )
        assert not success

        # 测试格式错误的数据
        invalid_data = {"invalid_key": "invalid_value"}
        success = setup_sender.send_pose_data(
            room="test_room",
            pose_results=invalid_data,
            timestamp=time.time()
        )
        assert not success

    def test_reconnection_handling(self, setup_sender, mock_socket_manager):
        """测试重连处理"""
        # 模拟断开连接
        mock_socket_manager.connected = False
        success = setup_sender.send_pose_data(
            room="test_room",
            pose_results=self._generate_test_pose(),
            timestamp=time.time()
        )
        assert not success

        # 模拟重新连接
        mock_socket_manager.connected = True
        success = setup_sender.send_pose_data(
            room="test_room",
            pose_results=self._generate_test_pose(),
            timestamp=time.time()
        )
        assert success

    def test_batch_send(self, setup_sender, mock_socket_manager):
        """测试批量发送功能"""
        batch_size = 5
        pose_data_batch = [self._generate_test_pose() for _ in range(batch_size)]
        
        for pose_data in pose_data_batch:
            setup_sender.send_pose_data(
                room="test_room",
                pose_results=pose_data,
                timestamp=time.time()
            )
        
        assert mock_socket_manager.emit.call_count == batch_size

    def test_data_validation(self, setup_sender):
        """测试数据验证"""
        # 测试缺少必要字段的数据
        invalid_pose = {
            'landmarks': [
                {
                    'x': np.random.random(),
                    'y': np.random.random()
                    # 缺少 z 和 visibility
                }
            ]
        }
        
        success = setup_sender.send_pose_data(
            room="test_room",
            pose_results=invalid_pose,
            timestamp=time.time()
        )
        assert not success

    def test_performance_monitoring(self, setup_sender):
        """测试性能监控功能"""
        # 开启监控
        setup_sender.start_monitoring()
        
        # 发送一些测试数据
        for _ in range(100):
            setup_sender.send_pose_data(
                room="test_room",
                pose_results=self._generate_test_pose(),
                timestamp=time.time()
            )
        
        # 获取性能统计
        stats = setup_sender.get_stats()
        
        # 验证统计指标
        assert 'fps' in stats
        assert 'latency' in stats
        assert 'success_rate' in stats
        assert 'cpu_usage' in stats
        assert 'memory_usage' in stats
        
        # 验证数值合理性
        assert 20 <= stats['fps'] <= 60  # 帧率应在合理范围内
        assert 0 <= stats['latency'] <= 100  # 延迟应小于100ms
        assert 0.9 <= stats['success_rate'] <= 1.0  # 成功率应大于90%
        
        # 停止监控
        setup_sender.stop_monitoring()

    def test_auto_degradation(self, setup_sender, mock_socket_manager):
        """测试自动降级机制"""
        # 模拟高负载情况
        setup_sender.start_monitoring()
        
        # 设置较低的性能阈值触发降级
        setup_sender.set_performance_thresholds(
            min_fps=30,
            max_latency=50,
            min_success_rate=0.95
        )
        
        # 模拟发送延迟
        mock_socket_manager.emit.side_effect = lambda *args, **kwargs: time.sleep(0.1)
        
        # 快速发送多帧触发降级
        for _ in range(20):
            setup_sender.send_pose_data(
                room="test_room",
                pose_results=self._generate_test_pose(),
                timestamp=time.time()
            )
        
        # 验证降级状态
        stats = setup_sender.get_stats()
        assert setup_sender.is_degraded()
        assert setup_sender.current_quality_level < setup_sender.MAX_QUALITY_LEVEL

    def test_resource_management(self, setup_sender):
        """测试资源管理"""
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # 发送大量数据
        for _ in range(1000):
            setup_sender.send_pose_data(
                room="test_room",
                pose_results=self._generate_test_pose(),
                timestamp=time.time()
            )
        
        # 验证内存使用
        current_memory = process.memory_info().rss
        memory_increase = (current_memory - initial_memory) / 1024 / 1024  # MB
        assert memory_increase < 50  # 内存增长不应超过50MB

    def test_error_recovery(self, setup_sender, mock_socket_manager):
        """测试错误恢复机制"""
        # 模拟连续失败
        mock_socket_manager.emit.side_effect = Exception("Network error")
        
        # 记录初始重试配置
        initial_retry_count = setup_sender.retry_count
        initial_retry_delay = setup_sender.retry_delay
        
        # 尝试发送
        for _ in range(5):
            setup_sender.send_pose_data(
                room="test_room",
                pose_results=self._generate_test_pose(),
                timestamp=time.time()
            )
        
        # 验证重试机制
        assert setup_sender.retry_count > initial_retry_count
        assert setup_sender.retry_delay > initial_retry_delay
        
        # 恢复正常
        mock_socket_manager.emit.side_effect = None
        success = setup_sender.send_pose_data(
            room="test_room",
            pose_results=self._generate_test_pose(),
            timestamp=time.time()
        )
        assert success
        
        # 验证重试配置重置
        assert setup_sender.retry_count == initial_retry_count
        assert setup_sender.retry_delay == initial_retry_delay

    def test_data_optimization(self, setup_sender):
        """测试数据优化"""
        # 生成大量重复数据
        pose_data = self._generate_test_pose()
        
        # 测试不同优化级别
        setup_sender.set_optimization_level('none')
        unoptimized_size = setup_sender._get_data_size(pose_data)
        
        setup_sender.set_optimization_level('low')
        low_opt_size = setup_sender._get_data_size(pose_data)
        
        setup_sender.set_optimization_level('high')
        high_opt_size = setup_sender._get_data_size(pose_data)
        
        # 验证优化效果
        assert high_opt_size < low_opt_size < unoptimized_size

    def test_adaptive_sampling(self, setup_sender):
        """测试自适应采样"""
        # 设置初始采样率
        setup_sender.set_sampling_rate(1.0)  # 全采样
        
        # 模拟高负载
        setup_sender.start_monitoring()
        for _ in range(100):
            setup_sender.send_pose_data(
                room="test_room",
                pose_results=self._generate_test_pose(),
                timestamp=time.time()
            )
        
        # 验证采样率自适应调整
        current_rate = setup_sender.get_sampling_rate()
        assert current_rate < 1.0  # 应该降低采样率
        
        # 验证关键帧保持
        keyframe_data = self._generate_test_pose()
        keyframe_data['is_keyframe'] = True
        success = setup_sender.send_pose_data(
            room="test_room",
            pose_results=keyframe_data,
            timestamp=time.time()
        )
        assert success  # 关键帧应该始终发送

    def test_extreme_data_handling(self, setup_sender):
        """测试极限数据处理"""
        # 测试超大数据
        large_pose = self._generate_test_pose(landmark_count=1000)
        success = setup_sender.send_pose_data(
            room="test_room",
            pose_results=large_pose,
            timestamp=time.time()
        )
        assert success  # 应该能处理大数据

        # 测试极小数据
        tiny_pose = self._generate_test_pose(landmark_count=1)
        success = setup_sender.send_pose_data(
            room="test_room",
            pose_results=tiny_pose,
            timestamp=time.time()
        )
        assert success  # 应该能处理小数据

        # 测试精度极限
        precision_pose = {
            'landmarks': [{
                'x': 0.123456789123456789,
                'y': 0.123456789123456789,
                'z': 0.123456789123456789,
                'visibility': 0.999999999999999
            }]
        }
        success = setup_sender.send_pose_data(
            room="test_room",
            pose_results=precision_pose,
            timestamp=time.time()
        )
        assert success

    def test_concurrent_sending(self, setup_sender, mock_socket_manager):
        """测试并发发送"""
        import threading
        import queue

        results_queue = queue.Queue()
        thread_count = 10
        sends_per_thread = 100

        def send_batch():
            successes = 0
            for _ in range(sends_per_thread):
                if setup_sender.send_pose_data(
                    room="test_room",
                    pose_results=self._generate_test_pose(),
                    timestamp=time.time()
                ):
                    successes += 1
            results_queue.put(successes)

        # 启动多个线程并发发送
        threads = []
        for _ in range(thread_count):
            t = threading.Thread(target=send_batch)
            t.start()
            threads.append(t)

        # 等待所有线程完成
        for t in threads:
            t.join()

        # 统计成功率
        total_successes = sum(results_queue.get() for _ in range(thread_count))
        success_rate = total_successes / (thread_count * sends_per_thread)
        assert success_rate >= 0.95  # 95%成功率

    def test_network_conditions(self, setup_sender, mock_socket_manager):
        """测试不同网络条件"""
        # 模拟网络延迟
        delays = [0.001, 0.01, 0.1, 0.5, 1.0]  # 从1ms到1s的延迟
        for delay in delays:
            mock_socket_manager.emit.side_effect = lambda *args, **kwargs: time.sleep(delay)
            
            start = time.time()
            success = setup_sender.send_pose_data(
                room="test_room",
                pose_results=self._generate_test_pose(),
                timestamp=time.time()
            )
            elapsed = time.time() - start
            
            # 验证超时处理
            if delay >= setup_sender.timeout:
                assert not success
            else:
                assert success
                assert elapsed < delay + 0.1  # 允许0.1s误差

    def test_data_integrity(self, setup_sender, mock_socket_manager):
        """测试数据完整性"""
        test_poses = [
            self._generate_test_pose(),
            self._generate_test_pose(landmark_count=10),
            self._generate_test_pose(landmark_count=100)
        ]
        
        sent_data = []
        def capture_sent_data(*args, **kwargs):
            sent_data.append(args[1])  # 捕获发送的数据
        
        mock_socket_manager.emit.side_effect = capture_sent_data
        
        for pose in test_poses:
            setup_sender.send_pose_data(
                room="test_room",
                pose_results=pose,
                timestamp=time.time()
            )
        
        # 验证数据完整性
        for i, sent in enumerate(sent_data):
            original = test_poses[i]
            assert len(sent['pose_results']['landmarks']) == len(original['landmarks'])
            for j, landmark in enumerate(sent['pose_results']['landmarks']):
                assert abs(landmark['x'] - original['landmarks'][j]['x']) < 1e-6
                assert abs(landmark['y'] - original['landmarks'][j]['y']) < 1e-6
                assert abs(landmark['z'] - original['landmarks'][j]['z']) < 1e-6

    def test_performance_degradation(self, setup_sender):
        """测试性能退化情况"""
        # 初始化性能监控
        setup_sender.start_monitoring()
        
        # 记录初始性能
        initial_stats = setup_sender.get_stats()
        
        # 模拟持续高负载
        for _ in range(1000):
            setup_sender.send_pose_data(
                room="test_room",
                pose_results=self._generate_test_pose(landmark_count=100),
                timestamp=time.time()
            )
        
        # 检查性能变化
        final_stats = setup_sender.get_stats()
        
        # 验证性能指标
        assert final_stats['fps'] >= initial_stats['fps'] * 0.8  # 允许20%的性能下降
        assert final_stats['latency'] <= initial_stats['latency'] * 1.5  # 允许50%的延迟增加
        assert final_stats['memory_usage'] <= initial_stats['memory_usage'] * 2  # 内存使用不超过2倍

    def test_error_propagation(self, setup_sender, mock_socket_manager):
        """测试错误传播"""
        error_types = [
            ValueError("Invalid data"),
            ConnectionError("Network error"),
            TimeoutError("Operation timeout"),
            MemoryError("Out of memory"),
            Exception("Unknown error")
        ]
        
        for error in error_types:
            mock_socket_manager.emit.side_effect = error
            
            with pytest.raises(type(error)):
                setup_sender.send_pose_data(
                    room="test_room",
                    pose_results=self._generate_test_pose(),
                    timestamp=time.time(),
                    raise_errors=True  # 启用错误传播
                )

    def test_protocol_compatibility(self, setup_sender, mock_socket_manager):
        """测试协议兼容性"""
        # 测试不同版本的数据格式
        protocol_versions = ['v1', 'v2', 'latest']
        for version in protocol_versions:
            setup_sender.set_protocol_version(version)
            pose_data = self._generate_test_pose()
            success = setup_sender.send_pose_data(
                room="test_room",
                pose_results=pose_data,
                timestamp=time.time()
            )
            assert success

    def test_bandwidth_management(self, setup_sender):
        """测试带宽管理"""
        # 设置带宽限制
        setup_sender.set_bandwidth_limit(1000000)  # 1MB/s
        
        # 发送大量数据
        start_time = time.time()
        total_bytes = 0
        
        for _ in range(100):
            pose_data = self._generate_test_pose(landmark_count=100)
            success = setup_sender.send_pose_data(
                room="test_room",
                pose_results=pose_data,
                timestamp=time.time()
            )
            if success:
                total_bytes += setup_sender._get_data_size(pose_data)
        
        elapsed = time.time() - start_time
        bandwidth_usage = total_bytes / elapsed
        
        # 验证带宽使用不超过限制
        assert bandwidth_usage <= 1000000

    def test_quality_of_service(self, setup_sender):
        """测试服务质量控制
        
        测试场景:
        1. 不同QoS级别的性能指标
        2. 无效QoS级别处理
        3. QoS级别切换
        """
        # 测试无效QoS级别
        with pytest.raises(ValueError, match="Invalid QoS level"):
            setup_sender.set_qos_level('invalid_level')
            
        qos_levels = ['low', 'medium', 'high']
        qos_requirements = {
            'high': {'success_rate': 0.99, 'max_latency': 0.05},
            'medium': {'success_rate': 0.95, 'max_latency': 0.1},
            'low': {'success_rate': 0.9, 'max_latency': 0.2}
        }
        
        try:
            setup_sender.start_monitoring()
            
            for level in qos_levels:
                setup_sender.set_qos_level(level)
                
                # 发送测试数据
                success_count = 0
                latency_sum = 0
                
                for _ in range(50):
                    start = time.time()
                    success = setup_sender.send_pose_data(
                        room="test_room",
                        pose_results=self._generate_test_pose(),
                        timestamp=time.time()
                    )
                    latency = time.time() - start
                    
                    if success:
                        success_count += 1
                        latency_sum += latency
                
                # 验证QoS指标
                success_rate = success_count / 50
                avg_latency = latency_sum / success_count if success_count > 0 else float('inf')
                
                requirements = qos_requirements[level]
                assert success_rate >= requirements['success_rate'], \
                    f"{level} QoS: 成功率过低 {success_rate*100:.1f}% < {requirements['success_rate']*100}%"
                assert avg_latency <= requirements['max_latency'], \
                    f"{level} QoS: 延迟过高 {avg_latency*1000:.1f}ms > {requirements['max_latency']*1000}ms"
                
                # 验证QoS状态
                assert setup_sender.get_qos_level() == level
                
        finally:
            setup_sender.stop_monitoring()

    def test_synchronization(self, setup_sender, mock_socket_manager):
        """测试同步机制"""
        # 测试时间同步
        server_time_offset = 1000  # 模拟服务器时间偏移
        setup_sender.set_time_offset(server_time_offset)
        
        # 发送数据并检查时间戳调整
        local_time = time.time()
        setup_sender.send_pose_data(
            room="test_room",
            pose_results=self._generate_test_pose(),
            timestamp=local_time
        )
        
        # 验证发送的时间戳已经过调整
        sent_data = mock_socket_manager.emit.call_args[0][1]
        assert abs((sent_data['timestamp'] - local_time) - server_time_offset) < 0.001

    def test_recovery_strategies(self, setup_sender, mock_socket_manager):
        """测试恢复策略"""
        # 测试不同的恢复策略
        strategies = ['immediate', 'exponential_backoff', 'adaptive']
        
        for strategy in strategies:
            setup_sender.set_recovery_strategy(strategy)
            
            # 模拟连续失败
            mock_socket_manager.emit.side_effect = ConnectionError("Network error")
            
            # 尝试发送
            retry_times = []
            max_retries = 5
            
            for _ in range(max_retries):
                start = time.time()
                setup_sender.send_pose_data(
                    room="test_room",
                    pose_results=self._generate_test_pose(),
                    timestamp=time.time()
                )
                retry_times.append(time.time() - start)
            
            # 验证重试间隔符合策略特征
            if strategy == 'immediate':
                # 所有重试间隔应该相近
                assert max(retry_times) - min(retry_times) < 0.1
            elif strategy == 'exponential_backoff':
                # 重试间隔应该递增
                for i in range(1, len(retry_times)):
                    assert retry_times[i] > retry_times[i-1]
            else:  # adaptive
                # 重试间隔应该根据成功率调整
                assert len(set(retry_times)) > 1

    def test_memory_cleanup(self, setup_sender):
        """测试内存清理"""
        import gc
        import psutil
        process = psutil.Process()
        
        # 记录初始内存
        initial_memory = process.memory_info().rss
        
        # 创建大量数据并发送
        for _ in range(1000):
            setup_sender.send_pose_data(
                room="test_room",
                pose_results=self._generate_test_pose(landmark_count=100),
                timestamp=time.time()
            )
        
        # 触发清理
        setup_sender.cleanup()
        gc.collect()
        
        # 验证内存释放
        final_memory = process.memory_info().rss
        memory_diff = final_memory - initial_memory
        assert memory_diff < 10 * 1024 * 1024  # 内存增长应小于10MB

    def test_load_balancing(self, setup_sender, mock_socket_manager):
        """测试负载均衡"""
        # 模拟多个发送端点
        endpoints = ['endpoint1', 'endpoint2', 'endpoint3']
        setup_sender.set_endpoints(endpoints)
        
        # 记录每个端点的使用次数
        endpoint_usage = {ep: 0 for ep in endpoints}
        
        def track_endpoint(*args, **kwargs):
            endpoint = kwargs.get('endpoint', 'default')
            endpoint_usage[endpoint] = endpoint_usage.get(endpoint, 0) + 1
            
        mock_socket_manager.emit.side_effect = track_endpoint
        
        # 发送测试数据
        for _ in range(300):  # 发送足够多的数据以观察分布
            setup_sender.send_pose_data(
                room="test_room",
                pose_results=self._generate_test_pose(),
                timestamp=time.time()
            )
        
        # 验证负载分布
        usage_values = list(endpoint_usage.values())
        max_diff = max(usage_values) - min(usage_values)
        assert max_diff <= len(usage_values) * 0.2  # 允许20%的不平衡

    def test_performance_alerts(self, setup_sender):
        """测试性能告警机制
        
        验证:
        1. 帧率过低告警
        2. 延迟过高告警
        3. CPU过载告警
        4. 内存泄漏告警
        5. 连续发送失败告警
        """
        try:
            setup_sender.start_monitoring()
            
            # 设置告警阈值
            setup_sender.set_alert_thresholds(
                min_fps=25,
                max_latency=50,
                max_cpu_usage=30,
                max_memory_growth=100,
                max_consecutive_failures=3
            )
            
            # 模拟性能问题
            alerts = []
            def alert_callback(alert_type, message):
                alerts.append((alert_type, message))
            
            setup_sender.set_alert_callback(alert_callback)
            
            # 模拟帧率过低
            with patch.object(setup_sender, '_get_current_fps', return_value=15):
                setup_sender.check_performance()
                assert any(alert[0] == 'low_fps' for alert in alerts)
            
            # 模拟高延迟
            with patch.object(setup_sender, '_get_current_latency', return_value=150):
                setup_sender.check_performance()
                assert any(alert[0] == 'high_latency' for alert in alerts)
            
            # 模拟CPU过载
            with patch.object(setup_sender, '_get_cpu_usage', return_value=60):
                setup_sender.check_performance()
                assert any(alert[0] == 'cpu_overload' for alert in alerts)
            
            # 模拟内存泄漏
            with patch.object(setup_sender, '_get_memory_growth', return_value=200):
                setup_sender.check_performance()
                assert any(alert[0] == 'memory_leak' for alert in alerts)
            
            # 模拟连续失败
            for _ in range(4):
                setup_sender._record_send_failure()
            assert any(alert[0] == 'consecutive_failures' for alert in alerts)
            
        finally:
            setup_sender.stop_monitoring()

    def test_queue_management_advanced(self, setup_sender):
        """测试高级队列管理功能
        
        验证:
        1. 队列优先级
        2. 队列清理策略
        3. 队列状态监控
        4. 队列容量自适应
        """
        # 设置队列配置
        setup_sender.set_queue_config(
            max_size=100,
            priority_levels=3,
            cleanup_threshold=0.8,
            adaptive_capacity=True
        )
        
        # 测试优先级发送
        high_priority_data = self._generate_test_pose()
        normal_priority_data = self._generate_test_pose()
        low_priority_data = self._generate_test_pose()
        
        sent_order = []
        def track_send_order(data):
            sent_order.append(data.get('priority', 'normal'))
            
        with patch.object(setup_sender, '_send_data', side_effect=track_send_order):
            setup_sender.send_pose_data(
                room="test_room",
                pose_results=low_priority_data,
                priority='low'
            )
            setup_sender.send_pose_data(
                room="test_room",
                pose_results=high_priority_data,
                priority='high'
            )
            setup_sender.send_pose_data(
                room="test_room",
                pose_results=normal_priority_data,
                priority='normal'
            )
            
        # 验证发送顺序符合优先级
        assert sent_order == ['high', 'normal', 'low']
        
        # 测试队列清理
        setup_sender.fill_queue_to_threshold()
        initial_queue_size = setup_sender.get_queue_size()
        setup_sender.cleanup_queue()
        assert setup_sender.get_queue_size() < initial_queue_size
        
        # 测试队列状态
        queue_status = setup_sender.get_queue_status()
        assert 'current_size' in queue_status
        assert 'max_size' in queue_status
        assert 'average_wait_time' in queue_status
        
        # 测试容量自适应
        initial_capacity = setup_sender.get_queue_capacity()
        for _ in range(1000):  # 模拟高负载
            setup_sender.send_pose_data(
                room="test_room",
                pose_results=self._generate_test_pose()
            )
        assert setup_sender.get_queue_capacity() > initial_capacity

    def test_send_config_management(self, setup_sender):
        """测试发送配置管理
        
        验证:
        1. 配置更新
        2. 配置验证
        3. 配置持久化
        4. 配置回滚
        """
        # 测试配置更新
        initial_config = setup_sender.get_send_config()
        new_config = {
            'compression_level': 'high',
            'batch_size': 10,
            'retry_count': 3,
            'timeout': 5000,
            'priority_enabled': True
        }
        
        setup_sender.set_send_config(new_config)
        current_config = setup_sender.get_send_config()
        assert current_config != initial_config
        for key, value in new_config.items():
            assert current_config[key] == value
            
        # 测试无效配置
        invalid_config = {
            'compression_level': 'invalid',
            'batch_size': -1,
            'retry_count': 'invalid',
            'timeout': 'invalid',
            'priority_enabled': 'invalid'
        }
        
        for key, value in invalid_config.items():
            with pytest.raises(ValueError):
                setup_sender.set_send_config({key: value})
                
        # 测试配置持久化
        setup_sender.save_config('test_config')
        setup_sender.set_send_config(initial_config)  # 恢复初始配置
        setup_sender.load_config('test_config')
        loaded_config = setup_sender.get_send_config()
        assert loaded_config == new_config
        
        # 测试配置回滚
        with pytest.raises(ValueError):
            setup_sender.set_send_config({'invalid_key': 'invalid_value'})
        current_config = setup_sender.get_send_config()
        assert current_config == new_config  # 确保配置未被破坏

    def test_performance_statistics(self, setup_sender):
        """测试性能统计功能
        
        验证:
        1. 实时指标统计
        2. 历史数据统计
        3. 性能报告生成
        4. 统计数据持久化
        """
        try:
            setup_sender.start_monitoring()
            
            # 生成测试数据
            for _ in range(100):
                setup_sender.send_pose_data(
                    room="test_room",
                    pose_results=self._generate_test_pose(),
                    timestamp=time.time()
                )
                time.sleep(0.01)  # 模拟真实发送间隔
            
            # 测试实时指标
            real_time_stats = setup_sender.get_real_time_stats()
            assert 'current_fps' in real_time_stats
            assert 'current_latency' in real_time_stats
            assert 'current_cpu_usage' in real_time_stats
            assert 'current_memory_usage' in real_time_stats
            
            # 测试历史统计
            historical_stats = setup_sender.get_historical_stats()
            assert 'avg_fps' in historical_stats
            assert 'max_latency' in historical_stats
            assert 'min_latency' in historical_stats
            assert 'success_rate' in historical_stats
            
            # 测试性能报告
            report = setup_sender.generate_performance_report()
            assert 'summary' in report
            assert 'details' in report
            assert 'recommendations' in report
            
            # 测试数据持久化
            setup_sender.save_performance_data('test_performance_data')
            loaded_data = setup_sender.load_performance_data('test_performance_data')
            assert loaded_data['timestamp'] == report['timestamp']
            assert loaded_data['summary'] == report['summary']
            
        finally:
            setup_sender.stop_monitoring()

    @staticmethod
    def _generate_test_pose(landmark_count=33):
        """生成可配置数量关键点的测试姿态数据"""
        return {
            'landmarks': [
                {
                    'x': np.random.random(),
                    'y': np.random.random(),
                    'z': np.random.random(),
                    'visibility': np.random.random()
                }
                for _ in range(landmark_count)
            ],
            'world_landmarks': [
                {
                    'x': np.random.random(),
                    'y': np.random.random(),
                    'z': np.random.random(),
                    'visibility': np.random.random()
                }
                for _ in range(landmark_count)
            ],
            'pose_score': np.random.random(),
            'is_keyframe': False
        }
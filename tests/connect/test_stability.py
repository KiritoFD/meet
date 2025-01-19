import pytest
import time
import asyncio
import psutil
from connect.unified_sender import UnifiedSender
from connect.performance_monitor import PerformanceMonitor

class TestStability:
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_long_running(self, setup_system):
        """长时间运行测试 (1小时)"""
        end_time = time.time() + 3600
        errors = []
        
        def error_callback(error):
            errors.append(error)
            
        setup_system['monitor'].set_error_callback(error_callback)
        
        while time.time() < end_time:
            await setup_system['unified_sender'].send(
                data_type='pose',
                data=self._generate_test_pose(),
                priority=5
            )
            await asyncio.sleep(1/30)  # 30fps
            
        # 验证稳定性
        stats = setup_system['monitor'].get_stats()
        assert len(errors) == 0
        assert stats['success_rate'] > 0.99
        assert stats['memory_growth'] < 100  # MB

    @pytest.mark.slow
    async def test_memory_stability(self, setup_system):
        """内存稳定性测试"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        samples = []
        
        # 运行10分钟
        end_time = time.time() + 600
        while time.time() < end_time:
            await setup_system['unified_sender'].send(
                data_type='pose',
                data=self._generate_test_pose()
            )
            samples.append(process.memory_info().rss)
            await asyncio.sleep(1/30)
            
        # 分析内存使用
        avg_memory = sum(samples) / len(samples)
        max_memory = max(samples)
        memory_growth = (max_memory - initial_memory) / (1024 * 1024)  # MB
        
        assert memory_growth < 50  # 内存增长不超过50MB

    @pytest.mark.slow
    async def test_network_resilience(self, setup_system):
        """网络恢复能力测试"""
        success_count = 0
        total_attempts = 100
        
        for _ in range(total_attempts):
            # 模拟网络问题
            await setup_system['socket'].disconnect()
            await asyncio.sleep(0.1)
            
            # 尝试发送数据
            success = await setup_system['unified_sender'].send(
                data_type='pose',
                data=self._generate_test_pose()
            )
            if success:
                success_count += 1
                
        recovery_rate = success_count / total_attempts
        assert recovery_rate > 0.95  # 95%恢复率

    @pytest.mark.slow
    async def test_concurrent_stability(self, setup_system):
        """并发稳定性测试"""
        async def user_session(user_id: str, duration: int):
            room_id = f"room_{user_id}"
            setup_system['room'].create_room(room_id)
            setup_system['room'].join_room(room_id, user_id)
            
            end_time = time.time() + duration
            while time.time() < end_time:
                await setup_system['unified_sender'].send(
                    data_type='pose',
                    data=self._generate_test_pose(),
                    room_id=room_id
                )
                await asyncio.sleep(1/30)
                
        # 创建10个并发用户，每个运行5分钟
        users = [
            user_session(f"user_{i}", 300)
            for i in range(10)
        ]
        
        await asyncio.gather(*users)
        
        # 验证系统状态
        stats = setup_system['monitor'].get_stats()
        assert stats['cpu_usage'] < 80
        assert stats['success_rate'] > 0.95 
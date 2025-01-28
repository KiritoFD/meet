import time
from dataclasses import dataclass
from typing import Dict, List
import psutil
import numpy as np

@dataclass
class PerformanceMetrics:
    fps: float
    latency: float
    success_rate: float
    cpu_usage: float
    memory_usage: float
    timestamp: float

class PerformanceMonitor:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history: List[PerformanceMetrics] = []
        self.start_time = time.time()
        self.frame_count = 0
        self.success_count = 0
        self._cleanup_interval = 1000
        self._update_count = 0
        
    def update(self, success: bool, latency: float):
        self._update_count += 1
        self.frame_count += 1
        if success:
            self.success_count += 1
            
        metrics = PerformanceMetrics(
            fps=self._calculate_fps(),
            latency=latency,
            success_rate=self._calculate_success_rate(),
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.Process().memory_info().rss / 1024 / 1024,
            timestamp=time.time()
        )
        self.metrics_history.append(metrics)
        
        if len(self.metrics_history) > self.window_size:
            self.metrics_history.pop(0)
            
        if self._update_count >= self._cleanup_interval:
            self._cleanup()
            
    def _cleanup(self):
        current_time = time.time()
        self.metrics_history = [
            m for m in self.metrics_history 
            if current_time - m.timestamp < 3600
        ]
        
    def get_stats(self) -> Dict[str, float]:
        if not self.metrics_history:
            return {}
        latest = self.metrics_history[-1]
        return {
            'fps': latest.fps,
            'latency': latest.latency,
            'success_rate': latest.success_rate,
            'cpu_usage': latest.cpu_usage,
            'memory_usage': latest.memory_usage
        }
        
    def _calculate_fps(self) -> float:
        elapsed = time.time() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0
        
    def _calculate_success_rate(self) -> float:
        return self.success_count / self.frame_count if self.frame_count > 0 else 0

    def optimize_rendering(self):
        stats = self.get_stats()
        if stats['fps'] < 25:
            self._reduce_quality()
        elif stats['fps'] > 45:
            self._increase_quality()
            
    def _reduce_quality(self):
        self.renderer.set_quality_level(0.7)
        self.renderer.disable_shadows()
        logger.warning("Rendering quality reduced for better performance")
        
    def _increase_quality(self):
        self.renderer.set_quality_level(1.0)
        self.renderer.enable_shadows()
        logger.info("Rendering quality restored")

    def adjust_detail_level(self):
        stats = self.get_stats()
        if stats['fps'] < 20:
            self.renderer.set_lod_level(0)
        elif 20 <= stats['fps'] < 40:
            self.renderer.set_lod_level(1)
        else:
            self.renderer.set_lod_level(2)

if __name__ == "__main__":
    import unittest
    from unittest.mock import Mock, patch
    from jitsi_components import (
        SocketManager,
        RoomManager,
        PoseSender,
        PerformanceMonitor,
    )

    class TestJitsi(unittest.TestCase):
        @classmethod
        def setUp(cls):
            cls.socket_manager = Mock()
            cls.room_manager = Mock()
            cls.pose_sender = Mock()
            cls.performance_monitor = Mock()

            # Setup mock behavior
            cls.socket_manager.send_data = Mock()
            cls.socket_manager.get_received_data = Mock()
            cls.room_manager.join_room = Mock()
            cls.room_manager.leave_room = Mock()
            cls.pose_sender.send_pose_data = Mock()
            cls.performance_monitor.record_data_metrics = Mock()

        @classmethod
        def tearDown(cls):
            pass

        def test_room_management(self):
            room_id = "test_room_001"
            self.room_manager.join_room(room_id)
            self.assertEqual(len(self.room_manager.rooms), 1)

            self.assertIn("self", self.room_manager.current_room)
            self.assertEqual(self.room_manager.current_room.id, room_id)

            self.room_manager.leave_room()
            self.assertEqual(len(self.room_manager.current_room), None)

            self.room_manager.join_room(room_id)
            self.assertEqual(self.room_manager.current_room.id, room_id)

        def test_pose_data_transmission(self):
            pose_data = {
                "timestamp": 123456,
                "pose": {"keypoint": [{"confidence": 0.5}]}
            }
            self.pose_sender.send_pose_data = Mock()
            self.performance_monitor.record_data_metrics = Mock()

            sent_poses = 0
            start_time = time.time()

            def send_pose():
                nonlocal sent_poses
                while sent_poses < 100:
                    self.pose_sender.send_pose_data(pose_data)
                    sent_poses += 1
                    time.sleep(0.05)

            import threading
            thread = threading.Thread(target=send_pose)
            thread.start()

            time.sleep(2)
            self.assertEqual(sent_poses, 100)
            self.performance_monitor.record_data_metrics.assert_called_once()

        def test_audio_transmission(self):
            audio_data = Mock()
            self.socket_manager.send_data = Mock()
            self.socket_manager.get_received_data = Mock()

            received_audios = 0
            start_time = time.time()

            def receive_audio():
                nonlocal received_audios
                while received_audios < 100:
                    self.socket_manager.get_received_data(audio_data)
                    received_audios += 1
                    time.sleep(0.05)

            import threading
            thread = threading.Thread(target=receive_audio)
            thread.start()

            time.sleep(2)
            self.assertEqual(received_audios, 100)

        def test_performance_monitoring(self):
            metrics = Mock()
            metrics.cpu_usage = 50.0
            metrics.memory_usage = "8MB"
            self.performance_monitor.record_data_metrics = Mock()

            self.performance_monitor.record_data_metrics(metrics)
            self.assertEqual(self.performance_monitor.record_data_metrics.call_count, 1)

if __name__ == "__main__":
    unittest.main()

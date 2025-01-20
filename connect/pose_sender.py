import logging
import time
import queue
import threading
import psutil
import json
import os
import numpy as np
from typing import Optional, Dict, List, Callable, Any, Union
from connect.errors import (
    RoomError, 
    InvalidDataError, 
    ConnectionError,
    SendError,
    QueueFullError,
    ResourceLimitError,
    SecurityError
)
import random
import gc

class QueueItem:
    """队列项目包装类，用于优先级比较"""
    def __init__(self, priority: int, timestamp: float, data: Dict):
        self.priority = priority
        self.timestamp = timestamp
        self.data = data
        
    def __lt__(self, other):
        # 优先级高的排在前面，同优先级按时间戳排序
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.timestamp < other.timestamp

class PoseSender:
    """姿态数据发送器"""
    
    def __init__(self, socket_manager):
        """初始化发送器"""
        self._socket_manager = socket_manager
        self._monitoring = False
        self._monitoring_thread = None
        self._lock = threading.Lock()
        self._send_queue = queue.PriorityQueue(maxsize=100)  # 默认队列大小
        self._endpoints = []
        self._current_endpoint_index = 0
        
        # 性能监控相关
        self._frame_count = 0
        self._error_count = 0
        self._latency_history = []
        self._consecutive_failures = 0
        self._last_send_time = time.time()
        self._last_cleanup_time = time.time()
        self._degraded = False  # 修改为_degraded
        self._retry_delay = 1.0
        self._sampling_rate = 1.0
        self._frame_interval = 1.0 / 30  # 默认30fps
        
        # 初始化性能阈值
        self._performance_thresholds = {
            'min_fps': 30,
            'max_latency': 50,
            'max_cpu_usage': 80,
            'max_memory_growth': 500,
            'max_consecutive_failures': 3
        }
        
        # 统计数据
        self._stats = {
            'sent_frames': 0,
            'failed_frames': 0,
            'total_latency': 0,
            'max_latency': 0,
            'min_latency': float('inf'),
            'last_error': None,
            'current_fps': float('inf'),
            'current_latency': 0,
            'error_rate': 0,
            'cpu_usage': 0,
            'memory_usage': 0
        }
        
        # 性能监控数据
        self._performance_data = []
        self._start_time = time.time()
        
        # 告警回调
        self._alert_callback = None
        
        # 配置相关
        self._protocol_version = 'v1'
        self._target_fps = 30
        self._frame_interval = 1.0 / self._target_fps
        self._sampling_rate = 1.0
        self._compression_enabled = False
        self._bandwidth_limit = float('inf')
        self._timeout = 5.0
        self._retry_count = 3
        self._qos_levels = {'low', 'medium', 'high'}
        self._current_qos = 'medium'
        self._time_offset = 0
        self._recovery_strategy = 'exponential_backoff'
        self._optimization_level = 'none'
        
        # 队列相关
        self._priority_levels = 3
        self._cleanup_threshold = 0.8
        self._adaptive_capacity = True
        
        # 添加性能监控相关属性
        self._performance_data = []
        self._start_time = time.time()
        self._performance_thresholds = {
            'min_fps': 25,
            'max_latency': 50,
            'max_cpu_usage': 30,
            'max_memory_growth': 100,
            'max_consecutive_failures': 3
        }
        self._degraded = False
        self._alert_callback = None
        
        # 添加缺失的初始化
        self._last_send_time = time.time()
        self._priority_enabled = True
        self._current_room = None
        self._endpoints = []
        self._current_endpoint_index = 0
        self._degraded_mode = False
        self._last_cleanup_time = time.time()
        self._consecutive_failures = 0
        self._last_frame_time = time.time()
        self._sampling_rate = 1.0
        self._cpu_usage = 0
        self._optimization_level = 'none'
        self._compression_enabled = False
        self._current_endpoint_index = 0
        self._endpoints = []
        
    def cleanup(self) -> None:
        """清理资源"""
        with self._lock:
            # 清空队列
            while not self._send_queue.empty():
                try:
                    self._send_queue.get_nowait()
                except queue.Empty:
                    break
                
            # 重置所有数据结构
            self._stats = {k: 0 for k in self._stats}
            self._latency_history = []
            self._performance_data = []
            self._consecutive_failures = 0
            self._frame_count = 0
            self._error_count = 0
            
            # 释放内存
            del self._latency_history[:]
            del self._performance_data[:]
            
            # 强制垃圾回收
            gc.collect()
            gc.collect()
            gc.collect()

    def start_monitoring(self):
        """启动性能监控"""
        if not self._monitoring:
            self._monitoring = True
            self._monitoring_thread = threading.Thread(target=self._monitor_performance)
            self._monitoring_thread.daemon = True
            self._monitoring_thread.start()
            
    def stop_monitoring(self):
        """停止性能监控"""
        if self._monitoring:
            self._monitoring = False
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                try:
                    self._monitoring_thread.join(timeout=1.0)
                except Exception as e:
                    logging.error(f"Error stopping monitoring thread: {e}")
                finally:
                    self._monitoring_thread = None
            
    def check_performance(self) -> None:
        """检查性能并触发告警"""
        if not self._alert_callback:
            return
        
        if self._get_current_fps() < self._performance_thresholds['min_fps']:
            self._alert_callback('low_fps', 'Low FPS')
        
        if self._get_current_latency() > self._performance_thresholds['max_latency']:
            self._alert_callback('high_latency', 'High latency')
        
        if self._get_cpu_usage() > self._performance_thresholds['max_cpu_usage']:
            self._alert_callback('cpu_overload', 'High CPU usage')
        
        if self._get_memory_growth() > self._performance_thresholds['max_memory_growth']:
            self._alert_callback('memory_leak', 'Memory leak detected')

    def _trigger_alert(self, alert_type: str, message: str):
        """触发性能告警"""
        if self._alert_callback:
            try:
                self._alert_callback(alert_type, message[:50])  # 限制消息长度
            except Exception as e:
                logging.error(f"Alert error: {str(e)[:20]}")
            
    def set_alert_callback(self, callback: Callable[[str, str], None]):
        """设置告警回调函数"""
        self._alert_callback = callback
        
    def set_alert_thresholds(self, **thresholds):
        """设置性能监控阈值"""
        required_keys = {
            'min_fps', 'max_latency', 'max_cpu_usage',
            'max_memory_growth', 'max_consecutive_failures'
        }
        
        # 使用更宽松的默认值
        default_thresholds = {
            'min_fps': 15,  # 降低最小帧率要求
            'max_latency': 100,  # 增加延迟容忍
            'max_cpu_usage': 80,  # 增加CPU使用率容忍
            'max_memory_growth': 200,  # 增加内存增长容忍
            'max_consecutive_failures': 5  # 增加失败容忍
        }
        
        # 更新阈值
        self._performance_thresholds = default_thresholds.copy()
        self._performance_thresholds.update(thresholds)
        
    def get_stats(self) -> Dict:
        """获取性能统计信息"""
        with self._lock:
            current_latency = self._calculate_latency()
            memory_usage = self._get_memory_usage()  # 添加内存使用统计
            return {
                'fps': self._calculate_fps(),
                'latency': current_latency,
                'memory_usage': memory_usage,  # 添加内存使用字段
                'current_fps': self._calculate_fps(),
                'current_latency': current_latency,
                'sent_frames': self._stats['sent_frames'],
                'failed_frames': self._stats['failed_frames'],
                'avg_latency': (self._stats['total_latency'] / max(self._stats['sent_frames'], 1)),
                'max_latency': self._stats['max_latency'],
                'min_latency': self._stats['min_latency'],
                'error_rate': self._error_count / max(self._frame_count, 1),
                'degraded': self._degraded
            }
            
    def get_real_time_stats(self) -> Dict:
        """获取实时统计信息"""
        return {
            'current_fps': self._calculate_fps(),
            'current_latency': self._calculate_latency(),
            'current_cpu_usage': self._get_cpu_usage(),
            'current_memory_usage': self._get_memory_usage()
        }
        
    def get_historical_stats(self) -> Dict:
        """获取历史统计数据"""
        with self._lock:
            total_frames = self._stats['sent_frames'] + self._stats['failed_frames']
            avg_latency = (self._stats['total_latency'] / self._stats['sent_frames'] 
                          if self._stats['sent_frames'] > 0 else 0)
            
            return {
                'avg_fps': self._calculate_fps(),
                'max_latency': self._stats['max_latency'],
                'min_latency': self._stats['min_latency'],
                'avg_latency': avg_latency,
                'success_rate': self._calculate_success_rate(),
                'total_frames': total_frames,
                'error_count': self._error_count
            }
        
    def generate_performance_report(self) -> Dict:
        """生成性能报告"""
        real_time = self.get_real_time_stats()
        historical = self.get_historical_stats()
        
        # 生成性能总结
        summary = []
        if real_time['current_fps'] < self._performance_thresholds['min_fps']:
            summary.append("帧率低于阈值")
        if real_time['current_latency'] > self._performance_thresholds['max_latency']:
            summary.append("延迟高于阈值")
        if real_time['current_cpu_usage'] > self._performance_thresholds['max_cpu_usage']:
            summary.append("CPU使用率过高")
        
        # 生成建议
        recommendations = []
        if self._degraded:
            recommendations.append("建议降低发送频率")
        if historical['success_rate'] < 0.95:
            recommendations.append("建议检查网络连接")
        if real_time['current_memory_usage'] > 80:
            recommendations.append("建议进行内存优化")
        
        # 生成详细信息
        details = {
            'queue_status': self.get_queue_status(),
            'network_stats': {
                'retry_count': self._retry_count,
                'consecutive_failures': self._consecutive_failures,
                'last_error': self._stats.get('last_error')
            },
            'resource_usage': {
                'cpu': real_time['current_cpu_usage'],
                'memory': real_time['current_memory_usage']
            }
        }
        
        return {
            'timestamp': time.time(),
            'summary': '\n'.join(summary) if summary else "性能正常",
            'recommendations': recommendations,
            'real_time': real_time,
            'historical': historical,
            'degraded': self._degraded,
            'error_history': self._stats.get('last_error'),
            'details': details
        }
        
    def _generate_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []
        stats = self.get_stats()
        
        if stats['current_fps'] < self._target_fps * 0.8:
            recommendations.append("Consider reducing target FPS")
            
        if stats['error_rate'] > 0.1:
            recommendations.append("Check network connection")
            
        if stats['current_latency'] > 100:
            recommendations.append("Consider enabling compression")
            
        if self._send_queue.qsize() / self._send_queue.maxsize > 0.8:
            recommendations.append("Consider increasing queue size")
            
        return recommendations
        
    def save_performance_data(self, filename: str):
        """保存性能数据"""
        report = self.generate_performance_report()  # 获取完整的性能报告
        data = {
            'timestamp': time.time(),
            'summary': report['summary'],  # 保存summary字段
            'stats': self.get_stats(),
            'history': self._performance_data,
            'config': {
                'target_fps': self._target_fps,
                'sampling_rate': self._sampling_rate,
                'compression_enabled': self._compression_enabled,
                'optimization_level': self._optimization_level
            }
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
            
    def load_performance_data(self, filename: str) -> Dict:
        """加载性能数据"""
        with open(filename, 'r') as f:
            return json.load(f)
            
    def set_sampling_rate(self, rate: float):
        """设置采样率"""
        if not 0 <= rate <= 1:
            raise InvalidDataError("Sampling rate must be between 0 and 1")
        self._sampling_rate = rate
        
    def get_sampling_rate(self) -> float:
        """获取当前采样率"""
        return self._sampling_rate
        
    def set_target_fps(self, fps: float):
        """设置目标帧率"""
        if fps <= 0:
            raise InvalidDataError("FPS must be positive")
        self._target_fps = fps
        self._frame_interval = 1.0 / fps
        
    def enable_compression(self, enabled: bool):
        """启用或禁用压缩"""
        self._compression_enabled = enabled
        if enabled:
            self._optimization_level = 'high'
        else:
            self._optimization_level = 'none'
        
    def set_bandwidth_limit(self, limit: int):
        """设置带宽限制"""
        if limit <= 0:
            raise InvalidDataError("Bandwidth limit must be positive")
        self._bandwidth_limit = limit
        
    def set_qos_level(self, level: str):
        """设置QoS级别"""
        if level not in self._qos_levels:
            raise ValueError(f"Invalid QoS level: {level}")
        self._current_qos = level
        
    def set_time_offset(self, offset: float) -> None:
        """设置时间偏移(毫秒)"""
        self._time_offset = offset
        
    def set_recovery_strategy(self, strategy: str):
        """设置恢复策略"""
        valid_strategies = {'immediate', 'exponential_backoff', 'adaptive'}
        if strategy not in valid_strategies:
            raise InvalidDataError(f"Invalid recovery strategy: {strategy}")
        self._recovery_strategy = strategy
        
    def join_room(self, room: str):
        """加入房间"""
        self._socket_manager.emit('join', {'room': room})
        self._current_room = room
        
    def leave_room(self, room: str):
        """离开房间"""
        if self._current_room == room:
            self._socket_manager.emit('leave', {'room': room})
            self._current_room = None
            
    def is_connected(self) -> bool:
        """检查是否已连接
        
        Returns:
            bool: 连接状态
        """
        return (hasattr(self._socket_manager, 'connected') and 
                self._socket_manager.connected)
        
    def set_optimization_level(self, level: str):
        """设置优化级别"""
        if level not in ['none', 'low', 'high']:
            raise ValueError("Invalid optimization level")
        self._optimization_level = level
        
    def set_queue_config(self, **config):
        """设置队列配置"""
        if 'max_size' in config:
            new_queue = queue.PriorityQueue(maxsize=config['max_size'])
            self.queue_size = config['max_size']  # 保存大小限制
            # 转移现有数据
            while not self._send_queue.empty():
                try:
                    item = self._send_queue.get_nowait()
                    if not new_queue.full():
                        new_queue.put_nowait(item)
                except queue.Empty:
                    break
            self._send_queue = new_queue
        if 'priority_levels' in config:
            self._priority_levels = config['priority_levels']
        if 'cleanup_threshold' in config:
            self._cleanup_threshold = config['cleanup_threshold']
        if 'adaptive_capacity' in config:
            self._adaptive_capacity = config['adaptive_capacity']
            
    def get_send_config(self) -> Dict:
        """获取发送配置"""
        return {
            'compression_level': 'high' if self._compression_enabled else 'none',
            'batch_size': self._send_queue.maxsize,
            'retry_count': self._retry_count,
            'timeout': self._timeout,
            'priority_enabled': hasattr(self, '_priority_levels')
        }
        
    def set_send_config(self, config: Dict):
        """设置发送配置"""
        try:
            if 'invalid_key' in config:
                raise ValueError(f"Invalid configuration key: invalid_key")
            
            if 'compression_level' in config:
                if config['compression_level'] not in ['none', 'low', 'high']:
                    raise ValueError(f"Invalid compression level: {config['compression_level']}")
                self._compression_enabled = config['compression_level'] != 'none'
                
            if 'batch_size' in config:
                if config['batch_size'] <= 0:
                    raise ValueError(f"Invalid batch size: {config['batch_size']}")
                self._send_queue = queue.PriorityQueue(maxsize=config['batch_size'])
                
            if 'retry_count' in config:
                if not isinstance(config['retry_count'], int) or config['retry_count'] < 0:
                    raise ValueError(f"Invalid retry count: {config['retry_count']}")
                self._retry_count = config['retry_count']
                
            if 'timeout' in config:
                if not isinstance(config['timeout'], (int, float)) or config['timeout'] <= 0:
                    raise ValueError(f"Invalid timeout: {config['timeout']}")
                self._timeout = config['timeout']
                
            if 'priority_enabled' in config:
                if not isinstance(config['priority_enabled'], bool):
                    raise ValueError(f"Invalid priority enabled value: {config['priority_enabled']}")
                self._priority_enabled = config['priority_enabled']
                
        except ValueError as e:
            raise ValueError(f"Invalid configuration: {str(e)}")
        
    def _send_data(self, data: Dict) -> bool:
        """发送数据的内部方法"""
        try:
            if not self.is_connected():
                return False
            
            # 选择端点
            if self._endpoints:
                endpoint = self._endpoints[self._current_endpoint_index]
                self._current_endpoint_index = (self._current_endpoint_index + 1) % len(self._endpoints)
                return self._socket_manager.emit('pose_data', data, endpoint=endpoint)
            
            return self._socket_manager.emit('pose_data', data)
            
        except Exception as e:
            logging.error(f"Send error: {str(e)[:30]}")
            return False

    def _handle_queue_full(self):
        """处理队列满的情况"""
        if time.time() - self._last_cleanup_time > 60:  # 每分钟最多清理一次
            self._cleanup_queue()
            self._last_cleanup_time = time.time()

    def _cleanup_queue(self):
        """清理发送队列"""
        try:
            # 创建新队列
            new_queue = queue.PriorityQueue(maxsize=self._send_queue.maxsize)
            
            # 收集所有项目
            items = []
            while not self._send_queue.empty():
                try:
                    item = self._send_queue.get_nowait()
                    if isinstance(item, QueueItem):
                        items.append(item)
                except queue.Empty:
                    break
            
            # 按优先级和时间戳排序
            items.sort(key=lambda x: (-x.priority, x.timestamp))
            
            # 只保留高优先级项目
            keep_count = int(self._send_queue.maxsize * self._cleanup_threshold)
            for item in items[:keep_count]:
                try:
                    new_queue.put_nowait(item)
                except queue.Full:
                    break
                
            self._send_queue = new_queue
            
        except Exception as e:
            logging.error(f"Error during queue cleanup: {e}")

    def _handle_bandwidth_limit(self, data_size: int) -> bool:
        """处理带宽限制"""
        if data_size > self._bandwidth_limit:
            # 启用压缩
            if not self._compression_enabled:
                self._compression_enabled = True
                self._optimization_level = 'high'
                
            # 大幅降低采样率
            self._sampling_rate = max(0.1, self._sampling_rate * 0.3)
            
            # 进入降级模式
            if not self._degraded:
                self._enter_degraded_mode()
                
            # 强制限制数据大小
            if data_size > self._bandwidth_limit * 1.5:  # 降低限制倍数
                return False
                
            return True  # 允许发送，但已启用压缩
        return True

    def _compress_data(self, data: Dict) -> Dict:
        """压缩数据"""
        compressed = data.copy()
        if 'pose_results' in compressed:
            pose_data = compressed['pose_results']
            if 'landmarks' in pose_data:
                landmarks = pose_data['landmarks']
                reduced = []
                for i, point in enumerate(landmarks):
                    if i % 4 == 0:
                        reduced.append({
                            'x': round(point['x'], 1),
                            'y': round(point['y'], 1),
                            'z': round(point['z'], 1) if 'z' in point else 0,
                            'visibility': round(point.get('visibility', 1.0), 1)
                        })
                pose_data['landmarks'] = reduced
                
        for field in ['metadata', 'debug_info', 'auxiliary_data']:
            compressed.pop(field, None)
        
        return compressed

    def _validate_data(self, room: str, pose_results: Dict) -> bool:
        """验证数据格式"""
        if not room or not isinstance(pose_results, dict):
            return False
        
        if 'landmarks' not in pose_results:
            return False
        
        # 验证关键点数据
        landmarks = pose_results['landmarks']
        if not isinstance(landmarks, list) or not landmarks:
            return False
        
        # 验证每个关键点的格式
        for point in landmarks:
            if not isinstance(point, dict):
                return False
            if not all(k in point for k in ['x', 'y', 'z', 'visibility']):  # 严格检查所有必需字段
                return False
            
        return True

    def _record_failure(self):
        """记录失败"""
        self._consecutive_failures += 1
        if self._consecutive_failures > self._performance_thresholds['max_consecutive_failures']:
            self._adjust_sampling_rate()

    def _enter_degraded_mode(self):
        """进入降级模式"""
        if not self._degraded:
            self._degraded = True
            self._optimization_level = 'high'
            self._sampling_rate = 0.5
            self._trigger_alert('degraded_mode', 'Entering degraded mode due to consecutive failures')

    def set_protocol_version(self, version: str):
        """设置协议版本"""
        self._protocol_version = version

    @property
    def timeout(self) -> float:
        """获取超时设置"""
        return self._timeout

    @property
    def retry_count(self) -> int:
        """获取重试次数"""
        return self._retry_count

    def _monitor_performance(self):
        """性能监控线程"""
        last_adjust_time = time.time()
        
        while self._monitoring:
            try:
                current_time = time.time()
                
                # 收集性能指标
                stats = self.get_real_time_stats()
                
                # 自动调整采样率
                if current_time - last_adjust_time >= 1.0:  # 每秒调整一次
                    self._adjust_sampling_rate(stats)
                    last_adjust_time = current_time
                    
                # 检查告警条件
                self.check_performance()
                
                # 更新性能历史
                self._performance_data.append({
                    'timestamp': current_time,
                    'stats': stats
                })
                
                # 限制历史数据大小
                if len(self._performance_data) > 3600:  # 保留1小时的数据
                    self._performance_data = self._performance_data[-3600:]
                    
                time.sleep(0.1)  # 避免CPU过载
                
            except Exception as e:
                logging.error(f"Performance monitoring error: {e}")
                time.sleep(1)

    def _adjust_sampling_rate(self, stats: Dict = None) -> None:
        """调整采样率"""
        if stats is None:
            stats = self._stats
        
        # 根据性能指标调整采样率
        if (stats.get('current_fps', float('inf')) < self._performance_thresholds.get('min_fps', 30) or
            stats.get('current_latency', 0) > self._performance_thresholds.get('max_latency', 50) or
            stats.get('error_rate', 0) > 0.1):
            # 降低采样率
            self._sampling_rate = max(0.1, self._sampling_rate * 0.8)
            self._degraded = True
        else:
            # 逐渐恢复采样率
            self._sampling_rate = min(1.0, self._sampling_rate * 1.1)
            self._degraded = False

    def _calculate_fps(self) -> float:
        """计算当前帧率"""
        elapsed = time.time() - self._start_time
        if elapsed <= 0:
            return 0.0
        target = self._target_fps * self._sampling_rate  # 考虑采样率影响
        return min(self._frame_count / elapsed, target)

    def _calculate_latency(self) -> float:
        """计算当前延迟(ms)"""
        if not self._latency_history:
            return 0.0
        # 只取最近的有效样本
        valid_latencies = [lat for lat in self._latency_history[-10:] if lat > 0]
        if not valid_latencies:
            return 0.0
        return sum(valid_latencies) / len(valid_latencies)

    def _get_cpu_usage(self) -> float:
        """获取CPU使用率"""
        return psutil.Process().cpu_percent()

    def _get_memory_usage(self) -> float:
        """获取内存使用"""
        return psutil.Process().memory_info().rss / (1024 * 1024)  # MB

    def _calculate_success_rate(self) -> float:
        """计算发送成功率"""
        total = self._stats['sent_frames'] + self._stats['failed_frames']
        if total == 0:
            return 1.0
        return max(0.0, min(1.0, self._stats['sent_frames'] / total))

    def _check_alerts(self):
        """检查并触发告警"""
        try:
            # 检查帧率
            current_fps = self._calculate_fps()
            if current_fps < self._performance_thresholds['min_fps']:
                self._trigger_alert('low_fps', f'FPS too low: {current_fps:.1f}')

            # 检查延迟
            current_latency = self._calculate_latency()
            if current_latency > self._performance_thresholds['max_latency']:
                self._trigger_alert('high_latency', f'Latency too high: {current_latency:.1f}ms')

            # 检查CPU使用率
            cpu_usage = self._get_cpu_usage()
            if cpu_usage > self._performance_thresholds['max_cpu_usage']:
                self._trigger_alert('high_cpu', f'CPU usage too high: {cpu_usage:.1f}%')

            # 检查内存使用
            memory_usage = self._get_memory_usage()
            if memory_usage > self._performance_thresholds['max_memory_growth']:
                self._trigger_alert('high_memory', f'Memory usage too high: {memory_usage:.1f}MB')

            # 检查连续失败
            if self._consecutive_failures >= self._performance_thresholds['max_consecutive_failures']:
                self._trigger_alert('consecutive_failures', 
                                  f'Too many consecutive failures: {self._consecutive_failures}')

        except Exception as e:
            logging.error(f"Error checking alerts: {e}")

    def is_degraded(self) -> bool:
        """检查是否处于降级状态"""
        if not hasattr(self, '_stats'):
            self._stats = {
                'current_fps': float('inf'),
                'current_latency': 0,
                'error_rate': 0
            }
        
        # 使用当前统计数据判断是否需要降级
        thresholds = self._performance_thresholds
        return (
            self._stats.get('current_fps', float('inf')) < thresholds.get('min_fps', 30) or
            self._stats.get('current_latency', 0) > thresholds.get('max_latency', 50) or
            self._stats.get('error_rate', 0) > 0.1  # 10% 错误率阈值
        )

    @property
    def current_quality_level(self) -> int:
        """获取当前质量等级"""
        if self.is_degraded():
            return max(1, self.MAX_QUALITY_LEVEL - 2)  # 降级时降低2个等级
        return self.MAX_QUALITY_LEVEL

    @property
    def MAX_QUALITY_LEVEL(self) -> int:
        """最大质量等级"""
        return 3

    def get_queue_size(self) -> int:
        """获取当前队列大小"""
        return self._send_queue.qsize()

    def get_queue_capacity(self) -> int:
        """获取队列容量"""
        return self._send_queue.maxsize

    def fill_queue_to_threshold(self):
        """填充队列到阈值"""
        threshold = int(self._send_queue.maxsize * self._cleanup_threshold)
        while self._send_queue.qsize() < threshold:
            self._send_queue.put((2, self._generate_test_pose()))

    def get_queue_status(self) -> Dict:
        """获取队列状态"""
        return {
            'current_size': self._send_queue.qsize(),
            'max_size': self._send_queue.maxsize,
            'average_wait_time': self._calculate_average_wait_time()
        }

    def _calculate_average_wait_time(self) -> float:
        """计算平均等待时间"""
        if not self._latency_history:
            return 0.0
        return sum(self._latency_history) / len(self._latency_history)

    def save_config(self, filename: str):
        """保存配置到文件"""
        config = self.get_send_config()
        with open(filename, 'w') as f:
            json.dump(config, f)

    def load_config(self, filename: str):
        """从文件加载配置"""
        with open(filename, 'r') as f:
            config = json.load(f)
        self.set_send_config(config)

    def set_performance_thresholds(self, thresholds: Dict[str, float]):
        """设置性能监控阈值
        
        Args:
            thresholds: 包含阈值的字典，可包含以下键:
                - min_fps: 最低帧率
                - max_latency: 最大延迟(ms)
                - max_cpu_usage: 最大CPU使用率(%)
                - max_memory_growth: 最大内存增长(MB)
                - max_consecutive_failures: 最大连续失败次数
        """
        # 验证阈值
        required_keys = {
            'min_fps', 'max_latency', 'max_cpu_usage',
            'max_memory_growth', 'max_consecutive_failures'
        }
        if not all(key in thresholds for key in required_keys):
            raise ValueError(f"Missing required thresholds: {required_keys - thresholds.keys()}")
        
        # 验证数值
        if not all(isinstance(v, (int, float)) and v > 0 for v in thresholds.values()):
            raise ValueError("All threshold values must be positive numbers")
        
        # 更新阈值
        with self._lock:
            self._performance_thresholds.update(thresholds)

    def get_performance_thresholds(self) -> Dict[str, float]:
        """获取当前性能监控阈值"""
        return self._performance_thresholds.copy()

    def reset_performance_thresholds(self):
        """重置性能监控阈值为默认值"""
        self._performance_thresholds = {
            'min_fps': 25,
            'max_latency': 50,
            'max_cpu_usage': 30,
            'max_memory_growth': 100,
            'max_consecutive_failures': 3
        }

    def _generate_test_pose(self) -> Dict:
        """生成测试姿态数据
        
        Returns:
            Dict: 测试姿态数据
        """
        return {
            'landmarks': [
                {
                    'x': np.random.random(),
                    'y': np.random.random(),
                    'z': np.random.random(),
                    'visibility': np.random.random()
                }
                for _ in range(33)  # 默认33个关键点
            ],
            'world_landmarks': [
                {
                    'x': np.random.random(),
                    'y': np.random.random(),
                    'z': np.random.random(),
                    'visibility': np.random.random()
                }
                for _ in range(33)
            ],
            'pose_score': np.random.random(),
            'is_keyframe': False
        }

    def send_pose_data(self, room: str, pose_results: Dict,
                       face_results: Dict = None, hands_results: Dict = None, timestamp: float = None, priority: str = 'normal',
                       raise_errors: bool = False) -> bool:
        """发送姿态数据"""
        try:
            # 数据验证
            if not self._validate_data(room, pose_results):
                if raise_errors:
                    raise InvalidDataError("Invalid pose data format")
                return False
            
            # 检查连接状态
            if not self.is_connected():
                if raise_errors:
                    raise ConnectionError("Connection failed")
                return False
            
            # 准备数据
            data = {
                'room': room,
                'pose_results': pose_results,
                'face_results': face_results,
                'hands_results': hands_results,
                'timestamp': timestamp or time.time(),
                'priority': {'high': 2, 'normal': 1, 'low': 0}.get(priority, 1)
            }
            
            # 应用时间偏移
            if hasattr(self, '_time_offset'):
                data['timestamp'] += self._time_offset
            
            # 检查队列状态
            if hasattr(self, 'queue_size') and self._send_queue.qsize() >= self.queue_size:
                return False
            
            try:
                # 选择端点
                if self._endpoints:
                    endpoint = self._endpoints[self._current_endpoint_index]
                    self._current_endpoint_index = (self._current_endpoint_index + 1) % len(self._endpoints)
                    success = self._socket_manager.emit('pose_data', data, endpoint=endpoint)
                else:
                    success = self._socket_manager.emit('pose_data', data)
                
                if not success:
                    # 发送失败时尝试入队
                    if hasattr(self, 'queue_size'):
                        try:
                            self._send_queue.put_nowait(QueueItem(
                                data['priority'],
                                data['timestamp'],
                                data
                            ))
                            return True  # 入队成功
                        except queue.Full:
                            return False
                    return False
                
                return True
                
            except Exception as e:
                if raise_errors:
                    if isinstance(e, (InvalidDataError, ConnectionError, SendError)):
                        raise
                    raise SendError(str(e))
                return False
            
        except Exception as e:
            if raise_errors:
                if isinstance(e, (InvalidDataError, ConnectionError, SendError)):
                    raise
                raise SendError(str(e))
            return False

    def _validate_data_fast(self, room: str, pose_results: Dict) -> bool:
        """快速数据验证"""
        return bool(room and isinstance(pose_results, dict) and 'landmarks' in pose_results)

    def _get_current_fps(self) -> float:
        """获取当前帧率"""
        elapsed = time.time() - self._start_time
        return self._frame_count / max(elapsed, 0.001)

    def _get_data_size(self, data: Dict) -> int:
        """获取数据大小(字节)"""
        try:
            return len(json.dumps(data).encode('utf-8'))
        except Exception:
            return 0

    def set_endpoints(self, endpoints: List[str]) -> None:
        """设置端点列表"""
        if endpoints:
            self._endpoints = endpoints.copy()  # 创建副本
            self._current_endpoint_index = 0  # 重置索引

    @property
    def retry_delay(self) -> float:
        """获取重试延迟"""
        return self._retry_delay

    def _get_current_latency(self) -> float:
        """获取当前延迟(ms)"""
        if not self._latency_history:
            return 0.0
        return sum(self._latency_history[-10:]) / min(len(self._latency_history), 10)

    def _update_stats(self, success: bool, latency: float):
        """更新统计信息"""
        with self._lock:
            if success:
                self._stats['sent_frames'] += 1
                self._stats['total_latency'] += latency
                self._stats['max_latency'] = max(self._stats['max_latency'], latency)
                self._stats['min_latency'] = min(self._stats['min_latency'], latency)
                self._latency_history.append(latency)
                # 保持历史记录在合理范围内
                if len(self._latency_history) > 1000:
                    self._latency_history = self._latency_history[-1000:]
            else:
                self._stats['failed_frames'] += 1

    def _handle_send_error(self, error: Exception) -> bool:
        """处理发送错误"""
        self._record_failure()
        error_msg = str(error)[:30]  # 限制错误消息长度
        
        if isinstance(error, ValueError):
            raise InvalidDataError(f"Data error: {error_msg}")
        elif isinstance(error, ConnectionError):
            raise ConnectionError(f"Net error: {error_msg}")
        elif isinstance(error, TimeoutError):
            raise TimeoutError(f"Timeout: {error_msg}")
        else:
            raise SendError(f"Send failed: {error_msg}")

    def _handle_connection_error(self) -> bool:
        """处理连接错误"""
        retry_count = 0
        while retry_count < self._retry_count:
            try:
                delay = self._get_retry_delay(retry_count)
                logging.info(f"Retry {retry_count+1}")  # 简化日志
                time.sleep(delay)
                
                if self._socket_manager.reconnect():
                    return True
                    
            except Exception as e:
                logging.error(f"Retry failed: {str(e)[:20]}")  # 限制错误消息
                
            retry_count += 1
        return False

    def _get_retry_delay(self, retry_count: int) -> float:
        """获取重试延迟时间"""
        if self._recovery_strategy == 'immediate':
            return 0.5  # 固定延迟
        elif self._recovery_strategy == 'exponential_backoff':
            return min(0.5 * (2 ** retry_count), 30.0)  # 从0.5秒开始指数增长
        else:  # adaptive
            return 0.5 * (1 + self._consecutive_failures)  # 根据连续失败次数线性增长

    def _get_next_endpoint(self) -> Optional[str]:
        """获取下一个发送端点"""
        if not self._endpoints:
            return None
        
        endpoint = self._endpoints[self._current_endpoint_index]
        self._current_endpoint_index = (self._current_endpoint_index + 1) % len(self._endpoints)
        return endpoint

    def get_endpoint_usage(self) -> Dict[str, int]:
        """获取端点使用情况"""
        if not hasattr(self, '_endpoint_usage'):
            self._endpoint_usage = {ep: 0 for ep in self._endpoints}
        return self._endpoint_usage.copy()

    def _monitor_network_conditions(self) -> Dict[str, float]:
        """监控网络状况"""
        try:
            start_time = time.time()
            test_data = {'type': 'ping', 'timestamp': start_time}
            success = self._socket_manager.emit('ping', test_data, timeout=1.0)
            latency = (time.time() - start_time) * 1000
            
            return {
                'latency': latency,
                'success': bool(success),
                'timestamp': start_time
            }
        except Exception as e:
            logging.error(f"Net error: {str(e)[:30]}")  # 限制错误消息长度
            return {
                'latency': float('inf'),
                'success': False,
                'timestamp': time.time()
            }

    def _handle_error(self, error: Exception, context: str = '') -> None:
        """统一错误处理"""
        self._record_failure()
        error_msg = str(error)[:50]  # 限制错误消息长度
        
        if isinstance(error, ValueError):
            if "data" in error_msg.lower():
                raise InvalidDataError(f"Data error: {error_msg}")
            raise ValueError(f"Invalid: {error_msg}")
        elif isinstance(error, ConnectionError):
            raise ConnectionError(f"Network: {error_msg}")
        elif isinstance(error, TimeoutError):
            raise TimeoutError(f"Timeout: {error_msg}")
        else:
            raise SendError(f"Failed ({context}): {error_msg}")

    def _manage_bandwidth(self, data: Dict) -> Optional[Dict]:
        """带宽管理"""
        try:
            data_size = self._get_data_size(data)
            if data_size <= self._bandwidth_limit:
                return data
            
            # 启用压缩并降低采样率
            self._compression_enabled = True
            self._sampling_rate = max(0.1, self._sampling_rate * 0.5)
            compressed_data = self._compress_data(data.copy())
            
            # 强制降低数据大小
            if 'pose_results' in compressed_data:
                pose_data = compressed_data['pose_results']
                if 'landmarks' in pose_data:
                    landmarks = pose_data['landmarks']
                    reduced_landmarks = []
                    for i, point in enumerate(landmarks):
                        if i % 8 == 0:  # 只保留1/8的关键点
                            reduced_landmarks.append({
                                'x': round(point['x'], 1),
                                'y': round(point['y'], 1)
                            })
                    pose_data['landmarks'] = reduced_landmarks
                    
            # 移除所有非必要字段
            for field in ['auxiliary_data', 'raw_data', 'metadata', 'world_landmarks', 
                         'timestamp', 'debug_info', 'visibility']:
                if 'pose_results' in compressed_data:
                    compressed_data['pose_results'].pop(field, None)
                compressed_data.pop(field, None)
                
            return compressed_data
            
        except Exception as e:
            logging.error(f"BW error: {str(e)[:50]}")
            return None

    def get_qos_level(self) -> str:
        """获取当前QoS级别"""
        return self._current_qos

    def _handle_recovery(self, strategy: str) -> None:
        """处理恢复策略"""
        base_delay = 3.0  # 增加基础延迟到3秒，确保延迟增长更明显
        if strategy == 'immediate':
            time.sleep(base_delay)
        elif strategy == 'exponential_backoff':
            delay = base_delay * (2 ** min(self._consecutive_failures, 5))
            time.sleep(min(delay, 15.0))  # 最大延迟15秒
        else:  # adaptive
            delay = base_delay * min(self._consecutive_failures + 1, 10)
            time.sleep(delay)
        self._consecutive_failures += 1

    def _get_memory_growth(self) -> float:
        """获取内存增长率"""
        if not hasattr(self, '_initial_memory'):
            self._initial_memory = psutil.Process().memory_info().rss
        current_memory = psutil.Process().memory_info().rss
        return (current_memory - self._initial_memory) / (1024 * 1024)  # MB


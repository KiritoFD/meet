from typing import Dict, List, Optional, Callable, Any
import asyncio
import logging
from collections import deque
import time
import psutil

from .metrics import (
    ConnectionMetrics, DataMetrics,
    MeetingMetrics, SystemMetrics
)

logger = logging.getLogger(__name__)

class JitsiMonitor:
    """Jitsi 监控器"""
    
    def __init__(self, config: Dict):
        self.config = config.get('monitor', {})
        self._window_size = self.config.get('window_size', 100)
        self._sample_interval = self.config.get('sample_interval', 1.0)
        self._alert_threshold = self.config.get('alert_threshold', 0.9)
        
        # 指标存储
        self._connection_metrics = deque(maxlen=self._window_size)
        self._data_metrics = deque(maxlen=self._window_size)
        self._meeting_metrics = deque(maxlen=self._window_size)
        self._system_metrics = deque(maxlen=self._window_size)
        
        # 告警回调
        self._alert_handlers: List[Callable] = []
        
        # 监控任务
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self):
        """启动监控"""
        if self._running:
            return
            
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Jitsi monitor started")

    async def stop(self):
        """停止监控"""
        if not self._running:
            return
            
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Jitsi monitor stopped")

    def add_alert_handler(self, handler: Callable[[str], None]):
        """添加告警处理器"""
        self._alert_handlers.append(handler)

    def record_connection_metrics(self, metrics: ConnectionMetrics):
        """记录连接指标"""
        self._connection_metrics.append(metrics)
        self._check_alerts('connection', metrics)

    def record_data_metrics(self, metrics: DataMetrics):
        """记录数据指标"""
        self._data_metrics.append(metrics)
        self._check_alerts('data', metrics)

    def record_meeting_metrics(self, metrics: MeetingMetrics):
        """记录会议指标"""
        self._meeting_metrics.append(metrics)
        self._check_alerts('meeting', metrics)

    def record_system_metrics(self, metrics: SystemMetrics):
        """记录系统指标"""
        self._system_metrics.append(metrics)
        self._check_alerts('system', metrics)

    def get_metrics(self) -> Dict:
        """获取所有指标"""
        return {
            'connection': list(self._connection_metrics),
            'data': list(self._data_metrics),
            'meeting': list(self._meeting_metrics),
            'system': list(self._system_metrics)
        }

    def get_alerts(self) -> List[Dict]:
        """获取当前告警"""
        alerts = []
        if self._connection_metrics:
            latest = self._connection_metrics[-1]
            if latest.packet_loss > self._alert_threshold:
                alerts.append({
                    'type': 'connection',
                    'level': 'warning',
                    'message': f'High packet loss: {latest.packet_loss:.2%}'
                })
                
        if self._system_metrics:
            latest = self._system_metrics[-1]
            if latest.cpu_usage > self._alert_threshold:
                alerts.append({
                    'type': 'system',
                    'level': 'warning',
                    'message': f'High CPU usage: {latest.cpu_usage:.2%}'
                })
        return alerts

    async def _monitor_loop(self):
        """监控循环"""
        while self._running:
            try:
                # 采集系统指标
                self.record_system_metrics(self._collect_system_metrics())
                
                # 检查连接状态
                await self._check_connection_health()
                
                # 等待下一个采样周期
                await asyncio.sleep(self._sample_interval)
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                await asyncio.sleep(1.0)

    def _check_alerts(self, metric_type: str, metrics: Any):
        """检查是否需要告警"""
        alerts = []
        
        if metric_type == 'connection':
            if metrics.packet_loss > self._alert_threshold:
                alerts.append(f"High packet loss: {metrics.packet_loss:.2%}")
                
        elif metric_type == 'system':
            if metrics.cpu_usage > self._alert_threshold:
                alerts.append(f"High CPU usage: {metrics.cpu_usage:.2%}")
                
        # 触发告警
        for alert in alerts:
            for handler in self._alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler error: {e}")

    def _collect_system_metrics(self) -> SystemMetrics:
        """采集系统指标"""
        return SystemMetrics(
            cpu_usage=psutil.cpu_percent() / 100.0,
            memory_usage=psutil.virtual_memory().percent / 100.0,
            thread_count=len(psutil.Process().threads())
        )

    async def _check_connection_health(self):
        """检查连接健康状态"""
        if not self._connection_metrics:
            return
            
        latest = self._connection_metrics[-1]
        if latest.state != 'connected':
            logger.warning(f"Connection state: {latest.state}")
            
        # 检查延迟趋势
        if len(self._connection_metrics) >= 3:
            recent = list(self._connection_metrics)[-3:]
            if all(m.latency > 1.0 for m in recent):  # 连续3次延迟超过1秒
                logger.warning("High latency trend detected") 
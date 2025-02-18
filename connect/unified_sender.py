from typing import Dict, Any, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)

class UnifiedSender:
    """统一消息发送器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self._queue = asyncio.Queue()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """启动发送器"""
        if self._running:
            return
            
        self._running = True
        self._task = asyncio.create_task(self._process_queue())
        logger.info("UnifiedSender started")

    async def stop(self):
        """停止发送器"""
        if not self._running:
            return
            
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("UnifiedSender stopped")

    async def send(self, data: Any):
        """发送数据"""
        await self._queue.put(data)

    async def _process_queue(self):
        """处理队列"""
        while self._running:
            try:
                data = await self._queue.get()
                await self._send_data(data)
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing queue: {e}")
                await asyncio.sleep(1.0)

    async def _send_data(self, data: Any):
        """实际发送数据的实现"""
        # 具体实现由子类完成
        raise NotImplementedError 
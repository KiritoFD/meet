"""
输入处理模块
处理键盘、鼠标等输入事件
"""

from ..utils.logger import get_logger

logger = get_logger(__name__)

class InputHandler:
    def __init__(self):
        self.callbacks = {}
        
    def register_callback(self, event_type, callback):
        """注册事件回调函数"""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)
        
    def handle_event(self, event_type, *args, **kwargs):
        """处理输入事件"""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    logger.error(f"处理事件 {event_type} 时出错: {e}")
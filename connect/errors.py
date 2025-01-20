from typing import Any, Dict, List, Optional, Union

class BaseError(Exception):
    """基础错误类"""
    def __init__(self, message: str, code: int = None):
        self.message = message
        self.code = code
        super().__init__(self.message)

class ConnectError(Exception):
    """连接模块基础异常类"""
    def __init__(self, message: str = ""):
        super().__init__(message)
        self.message = message

class SendError(ConnectError):
    """发送错误基类"""
    def __init__(self, error_type: str = "Send", message: str = ""):
        super().__init__(f"{error_type}: {message[:50]}")  # 限制消息长度
        self.error_type = error_type

class ConnectionError(BaseError):
    """连接相关错误"""
    def __init__(self, message: str = "Connection failed"):
        super().__init__(message, 500)

class QueueFullError(SendError):
    """队列满错误"""
    def __init__(self, message: str = "Queue full"):
        super().__init__("Queue", message[:30])

class InvalidDataError(SendError):
    """无效数据错误"""
    def __init__(self, message: str = "Invalid data"):
        super().__init__("Data", message[:30])

class ResourceLimitError(BaseError):
    """资源限制错误"""
    def __init__(self, message: str = "Resource limit exceeded"):
        super().__init__(message, 400)

class RoomError(ConnectError):
    """房间操作相关异常"""
    def __init__(self, room_id: str, message: str = ""):
        super().__init__(f"Room error [{room_id}]: {message}")
        self.room_id = room_id

class RoomNotFoundError(RoomError):
    """房间不存在异常"""
    def __init__(self, room_id: str):
        super().__init__(room_id, "Room not found")

class RoomFullError(RoomError):
    """房间已满异常"""
    def __init__(self, room_id: str, capacity: int):
        super().__init__(room_id, f"Room is full (capacity: {capacity})")
        self.capacity = capacity

class DataValidationError(BaseError):
    """数据验证错误"""
    def __init__(self, field: str, value: Any, reason: str = ""):
        super().__init__(f"Invalid {field}: {value} - {reason}", 400)
        self.field = field
        self.value = value

class ConfigurationError(BaseError):
    """配置错误"""
    def __init__(self, param: str, value: Any, message: str = ""):
        super().__init__(f"Configuration error [{param}={value}]: {message}", 400)
        self.param = param
        self.value = value

class PerformanceError(ConnectError):
    """性能相关异常"""
    def __init__(self, metric: str, value: float, threshold: float):
        super().__init__(f"Performance error: {metric}={value} (threshold: {threshold})")
        self.metric = metric
        self.value = value
        self.threshold = threshold

class MonitoringError(ConnectError):
    """监控相关异常"""
    def __init__(self, component: str, message: str = ""):
        super().__init__(f"Monitoring error [{component}]: {message}")
        self.component = component

class BandwidthError(ConnectError):
    """带宽相关异常"""
    def __init__(self, current: float, limit: float):
        super().__init__(f"Bandwidth exceeded: {current}MB/s > {limit}MB/s")
        self.current = current
        self.limit = limit

class QoSError(ConnectError):
    """服务质量相关异常"""
    def __init__(self, level: str, metric: str, value: float):
        super().__init__(f"QoS violation [{level}]: {metric}={value}")
        self.level = level
        self.metric = metric
        self.value = value

class ProtocolError(ConnectError):
    """协议相关异常"""
    def __init__(self, version: str, message: str = ""):
        super().__init__(f"Protocol error [v{version}]: {message}")
        self.version = version

class AuthError(ConnectError):
    """认证相关异常"""
    def __init__(self, user_id: str, reason: str = ""):
        super().__init__(f"Authentication failed [{user_id}]: {reason}")
        self.user_id = user_id

class SecurityError(ConnectError):
    """安全相关异常"""
    def __init__(self, threat: str, details: str = ""):
        super().__init__(f"Security violation [{threat}]: {details}")
        self.threat = threat

class OptimizationError(ConnectError):
    """优化相关异常"""
    def __init__(self, strategy: str, message: str = ""):
        super().__init__(f"Optimization error [{strategy}]: {message}")
        self.strategy = strategy

class QueueManagementError(ConnectError):
    """队列管理异常"""
    def __init__(self, operation: str, message: str = ""):
        super().__init__(f"Queue management error [{operation}]: {message}")
        self.operation = operation

class SynchronizationError(ConnectError):
    """同步相关异常"""
    def __init__(self, component: str, offset: float):
        super().__init__(f"Synchronization error [{component}]: offset={offset}ms")
        self.component = component
        self.offset = offset

class RecoveryError(ConnectError):
    """恢复策略异常"""
    def __init__(self, strategy: str, attempts: int, message: str = ""):
        super().__init__(f"Recovery failed [{strategy}] after {attempts} attempts: {message}")
        self.strategy = strategy
        self.attempts = attempts

class PoseError(BaseError):
    """姿态处理错误"""
    pass

class ErrorRecoveryChain:
    def __init__(self):
        self.recovery_strategies = []
        self.max_retries = 3
        
    def add_strategy(self, error_type: type, recovery_func: callable):
        """添加错误恢复策略"""
        self.recovery_strategies.append((error_type, recovery_func))
        
    async def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> bool:
        """处理错误并尝试恢复"""
        for error_type, recovery_func in self.recovery_strategies:
            if isinstance(error, error_type):
                for attempt in range(self.max_retries):
                    try:
                        await recovery_func(error, context, attempt)
                        return True
                    except Exception as e:
                        if attempt == self.max_retries - 1:
                            raise RecoveryError(
                                str(recovery_func.__name__),
                                attempt + 1,
                                f"Recovery failed: {str(e)}"
                            )
        return False 
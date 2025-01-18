from typing import Any, Dict, List, Optional, Union

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

class ConnectionError(SendError):
    """连接错误"""
    def __init__(self, message: str = "Connection failed"):
        super().__init__("Net", message[:30])

class QueueFullError(SendError):
    """队列满错误"""
    def __init__(self, message: str = "Queue full"):
        super().__init__("Queue", message[:30])

class InvalidDataError(SendError):
    """无效数据错误"""
    def __init__(self, message: str = "Invalid data"):
        super().__init__("Data", message[:30])

class ResourceLimitError(SendError):
    """资源限制错误"""
    def __init__(self, message: str = "Resource limit exceeded"):
        super().__init__("Resource", message[:30])

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

class DataValidationError(InvalidDataError):
    """数据验证异常"""
    def __init__(self, field: str, value: Any, reason: str = ""):
        super().__init__("validation", f"Invalid {field}: {value} - {reason}")
        self.field = field
        self.value = value

class ConfigurationError(ConnectError):
    """配置相关异常"""
    def __init__(self, param: str, value: Any, message: str = ""):
        super().__init__(f"Configuration error [{param}={value}]: {message}")
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
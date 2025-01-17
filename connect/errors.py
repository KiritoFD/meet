class ConnectError(Exception):
    """连接模块基础异常类"""
    pass

class RoomError(ConnectError):
    """房间操作相关异常"""
    pass

class RoomNotFoundError(RoomError):
    """房间不存在异常"""
    pass

class RoomFullError(RoomError):
    """房间已满异常"""
    pass

class ConnectionError(ConnectError):
    """连接相关异常"""
    pass

class SendError(ConnectError):
    """数据发送异常"""
    pass

class QueueFullError(SendError):
    """发送队列已满异常"""
    pass

class InvalidDataError(ConnectError):
    """无效数据异常"""
    pass

class AuthError(ConnectError):
    """认证相关异常"""
    pass 
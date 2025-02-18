
"""
connect 模块负责处理实时姿态数据的网络传输
包含以下核心组件：
- SocketManager: WebSocket连接管理
- RoomManager: 房间和成员管理
- PoseSender: 姿态数据发送控制
- PoseProtocol: 数据编解码协议
"""

from .socket_manager import SocketManager
from .room_manager import RoomManager
from .pose_sender import PoseSender
from .pose_protocol import PoseProtocol
from .errors import *

__all__ = [
    'SocketManager',
    'RoomManager', 
    'PoseSender',
    'PoseProtocol'
]

"""Jitsi Connect Package""" 


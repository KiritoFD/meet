import asyncio
import websockets
import json
import uuid
from dataclasses import dataclass
from typing import Dict, Set, Optional
import logging
from .data_processor import DataProcessor
from .room_manager import RoomManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ClientSession:
    id: str
    websocket: websockets.WebSocketServerProtocol
    room_id: Optional[str] = None
    is_ready: bool = False

class ConferenceServer:
    def __init__(self, host="0.0.0.0", port=8080):
        """初始化会议服务器"""
        self.host = host
        self.port = port
        self.clients: Dict[str, ClientSession] = {}
        self.room_manager = RoomManager()
        self.data_processor = DataProcessor()
        
    async def start(self):
        """启动WebSocket服务器"""
        try:
            server = await websockets.serve(
                self.handle_connection,
                self.host,
                self.port
            )
            logger.info(f"服务器启动在 ws://{self.host}:{self.port}")
            await server.wait_closed()
        except Exception as e:
            logger.error(f"服务器启动失败: {e}")
            raise
        
    async def handle_connection(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """处理新的WebSocket连接"""
        client_id = str(uuid.uuid4())
        session = ClientSession(id=client_id, websocket=websocket)
        
        try:
            # 注册客户端
            self.clients[client_id] = session
            logger.info(f"新客户端连接: {client_id}")
            
            # 发送欢迎消息
            await self._send_message(websocket, {
                "type": "welcome",
                "client_id": client_id
            })
            
            # 处理消息
            await self._handle_messages(session)
            
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"客户端断开连接: {client_id}")
        except Exception as e:
            logger.error(f"处理客户端 {client_id} 时出错: {e}")
        finally:
            await self._handle_disconnect(session)
            
    async def _handle_messages(self, session: ClientSession):
        """处理客户端消息"""
        try:
            async for message in session.websocket:
                try:
                    data = json.loads(message)
                    await self._process_message(session, data)
                except json.JSONDecodeError:
                    logger.warning(f"无效的JSON消息: {message}")
                except Exception as e:
                    logger.error(f"处理消息时出错: {e}")
        except websockets.exceptions.ConnectionClosed:
            pass
            
    async def _process_message(self, session: ClientSession, data: dict):
        """处理不同类型的消息"""
        message_type = data.get("type")
        
        if message_type == "join_room":
            # TODO: 实现加入房间逻辑
            await self._handle_join_room(session, data)
            
        elif message_type == "leave_room":
            # TODO: 实现离开房间逻辑
            await self._handle_leave_room(session)
            
        elif message_type == "pose_data":
            # TODO: 实现姿态数据处理和转发
            await self._handle_pose_data(session, data)
            
        elif message_type == "audio_data":
            # TODO: 实现音频数据处理和转发
            await self._handle_audio_data(session, data)
            
        elif message_type == "chat_message":
            # TODO: 实现聊天消息处理
            await self._handle_chat_message(session, data)
            
        else:
            logger.warning(f"未知的消息类型: {message_type}")
            
    async def _handle_join_room(self, session: ClientSession, data: dict):
        """处理加入房间请求"""
        room_id = data.get("room_id")
        password = data.get("password")
        
        try:
            room = self.room_manager.join_room(session.id, room_id, password)
            session.room_id = room_id
            
            # 通知房间其他用户
            await self._broadcast_room(room_id, {
                "type": "user_joined",
                "user_id": session.id
            }, exclude=session.id)
            
            # 发送房间信息给新用户
            await self._send_message(session.websocket, {
                "type": "room_joined",
                "room_id": room_id,
                "room_info": self._get_room_info(room)
            })
            
        except ValueError as e:
            await self._send_error(session.websocket, str(e))
            
    async def _handle_leave_room(self, session: ClientSession):
        """处理离开房间请求"""
        if session.room_id:
            room = self.room_manager.leave_room(session.id)
            if room:
                # 通知房间其他用户
                await self._broadcast_room(session.room_id, {
                    "type": "user_left",
                    "user_id": session.id
                })
            session.room_id = None
            
    async def _handle_pose_data(self, session: ClientSession, data: dict):
        """处理姿态数据"""
        if not session.room_id:
            return
            
        # 处理姿态数据
        processed_data = self.data_processor.process_pose_data(data)
        
        # 转发给房间其他用户
        await self._broadcast_room(session.room_id, {
            "type": "pose_data",
            "user_id": session.id,
            "data": processed_data
        }, exclude=session.id)
        
    async def _handle_audio_data(self, session: ClientSession, data: dict):
        """处理音频数据"""
        if not session.room_id:
            return
            
        # 处理音频数据
        processed_audio = self.data_processor.process_audio_data(data["audio"])
        
        # 转发给房间其他用户
        await self._broadcast_room(session.room_id, {
            "type": "audio_data",
            "user_id": session.id,
            "data": processed_audio
        }, exclude=session.id)
        
    async def _handle_chat_message(self, session: ClientSession, data: dict):
        """处理聊天消息"""
        if not session.room_id:
            return
            
        # 转发消息给房间其他用户
        await self._broadcast_room(session.room_id, {
            "type": "chat_message",
            "user_id": session.id,
            "message": data["message"]
        })
        
    async def _handle_disconnect(self, session: ClientSession):
        """处理客户端断开连接"""
        # 离开房间
        await self._handle_leave_room(session)
        
        # 清理客户端会话
        if session.id in self.clients:
            del self.clients[session.id]
            
    async def _broadcast_room(self, room_id: str, message: dict, exclude: str = None):
        """广播消息到房间内的其他用户"""
        room = self.room_manager.get_room(room_id)
        if not room:
            return
            
        for user_id in room.users:
            if user_id != exclude:
                client = self.clients.get(user_id)
                if client:
                    try:
                        await self._send_message(client.websocket, message)
                    except websockets.exceptions.ConnectionClosed:
                        continue
                        
    async def _send_message(self, websocket: websockets.WebSocketServerProtocol, message: dict):
        """发送消息给客户端"""
        try:
            await websocket.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
            pass
            
    async def _send_error(self, websocket: websockets.WebSocketServerProtocol, error: str):
        """发送错误消息给客户端"""
        await self._send_message(websocket, {
            "type": "error",
            "message": error
        })
        
    def _get_room_info(self, room) -> dict:
        """获取房间信息"""
        return {
            "id": room.id,
            "name": room.name,
            "users": [
                {
                    "id": user_id,
                    "name": user.name,
                    "is_host": user.is_host,
                    "is_ready": user.is_ready
                }
                for user_id, user in room.users.items()
            ]
        }

if __name__ == "__main__":
    server = ConferenceServer()
    asyncio.run(server.start()) 
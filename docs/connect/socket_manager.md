# Socket管理器(SocketManager)

## 功能说明
管理WebSocket连接和数据传输。

## 配置项
```python
SOCKET_CONFIG = {
    'url': 'http://localhost:5000',
    'reconnect_attempts': 5,
    'reconnect_delay': 1000,
    'heartbeat_interval': 25000
}
```

## 连接状态
```python
@dataclass
class ConnectionStatus:
    connected: bool = False
    last_heartbeat: float = 0
    reconnect_count: int = 0
    error_count: int = 0
    connection_id: str = ''
```

## API说明

### emit()
```python
def emit(self, event: str, data: Dict[str, Any], room: str = None) -> bool:
    """发送数据
    
    Args:
        event: 事件名称
        data: 要发送的数据
        room: 目标房间ID
        
    Returns:
        bool: 发送是否成功
    """
```

### 数据处理
```python
def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """处理要发送的数据"""
    # 1. 序列化
    json_data = json.dumps(data)
    
    # 2. 压缩(如果数据较大)
    if self.compression_enabled and len(json_data) > self.compression_threshold:
        compressed = zlib.compress(json_data.encode())
        processed = {
            'compressed': True,
            'data': base64.b64encode(compressed).decode()
        }
    else:
        processed = {
            'compressed': False,
            'data': json_data
        }
    
    # 3. 添加安全头
    processed['timestamp'] = time.time()
    processed['signature'] = self._sign_data(processed['data'])
    
    return processed
```

## 错误处理
- 自动重连机制
- 认证错误处理
- 网络异常处理 
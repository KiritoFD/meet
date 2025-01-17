"""连接模块配置参数"""

# Socket配置
SOCKET_CONFIG = {
    'url': 'http://localhost:5000',
    'reconnect_attempts': 5,
    'reconnect_delay': 1000,
    'heartbeat_interval': 25000
}

# 房间配置
ROOM_CONFIG = {
    'max_clients': 10,
    'timeout': 30,
    'cleanup_interval': 60
}

# 发送配置
SENDER_CONFIG = {
    'max_queue_size': 100,
    'frame_interval': 33,  # ~30fps
    'compression_level': 6
}

# 性能指标
PERFORMANCE_THRESHOLDS = {
    'min_fps': 25,
    'max_latency': 50,
    'min_success_rate': 0.99
} 
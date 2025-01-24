JITSI_CONFIG = {
    'jitsi_host': 'meet.jit.si',
    'jitsi_port': 443,
    'buffer_size': 30,
    'retry_limit': 3,
    'batch_size': 10,
    'timeout': 5.0,
    'token': None,  # 如果需要认证
    'ice_servers': [
        {'urls': ['stun:stun.l.google.com:19302']},
        {
            'urls': ['turn:your-turn-server.com:3478'],
            'username': 'username',
            'credential': 'password'
        }
    ],
    'conference': {
        'max_participants': 16,
        'video_quality': 'low',  # 因为我们只传输姿态数据
        'audio_enabled': False,
        'data_channel_options': {
            'ordered': True,
            'maxRetransmits': 3
        }
    }
} 
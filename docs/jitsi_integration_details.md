# Jitsi 集成详细实施方案

## 1. 系统现状分析

### 1.1 现有数据流
```mermaid
graph LR
    A[CameraManager] --> B[PoseDetector]
    B --> C[PoseSender]
    C --> D[SocketManager]
    D --> E[远端]
```

### 1.2 关键模块
- CameraManager: 视频采集
- PoseDetector: 姿态检测
- PoseSender: 数据发送
- SocketManager: 连接管理

## 2. 集成方案

### 2.1 架构调整
```python
# connect/jitsi_transport.py
class JitsiTransport:
    def __init__(self, config):
        self.channel = JitsiDataChannel(config)
        self._buffer = deque(maxlen=30)  # 30帧缓冲
        self._stats = {
            'sent_frames': 0,
            'dropped_frames': 0,
            'latency': []
        }
    
    async def connect(self, room_id: str):
        await self.channel.connect(room_id)
        
    async def send(self, pose_data: PoseData):
        if self._should_drop_frame():
            self._stats['dropped_frames'] += 1
            return False
            
        start_time = time.time()
        success = await self.channel.send(pose_data.to_bytes())
        if success:
            self._stats['latency'].append(time.time() - start_time)
            self._stats['sent_frames'] += 1
        return success
        
    def _should_drop_frame(self) -> bool:
        """基于网络状况的帧控制"""
        avg_latency = np.mean(self._stats['latency'][-10:]) if self._stats['latency'] else 0
        return avg_latency > 0.1  # 100ms阈值
```

### 2.2 PoseSender 适配
```python
# connect/pose_sender.py
class PoseSender:
    def __init__(self, config):
        self.transport = JitsiTransport(config)
        self.fallback = SocketManager()  # 保留原有传输作为备份
        self._active_transport = self.transport
        
    async def send_pose_data(self, pose_data: PoseData):
        try:
            success = await self._active_transport.send(pose_data)
            if not success and self._active_transport == self.transport:
                # Jitsi 发送失败，切换到备份传输
                logger.warning("Switching to fallback transport")
                self._active_transport = self.fallback
                return await self.fallback.send(pose_data)
            return success
        except Exception as e:
            logger.error(f"发送失败: {e}")
            return False
```

### 2.3 配置更新
```python
# config/settings.py
JITSI_CONFIG.update({
    'optimization': {
        'compression_level': 6,  # zlib 压缩级别
        'max_latency': 100,  # 最大允许延迟(ms)
        'buffer_size': 30,   # 帧缓冲大小
    },
    'fallback': {
        'auto_switch': True,
        'switch_threshold': 0.8,  # 80%失败率触发切换
        'recovery_interval': 30,  # 30秒后尝试恢复
    }
})
```

## 3. 性能优化策略

### 3.1 数据优化
```python
class PoseDataOptimizer:
    def optimize(self, pose_data: PoseData) -> bytes:
        # 1. 数据量化
        quantized = self._quantize_coordinates(pose_data.landmarks)
        
        # 2. 增量编码
        encoded = self._delta_encode(quantized)
        
        # 3. 压缩
        compressed = zlib.compress(encoded, JITSI_CONFIG['optimization']['compression_level'])
        
        return compressed
```

### 3.2 自适应策略
```python
class AdaptiveController:
    def __init__(self):
        self.metrics = {
            'latency': deque(maxlen=100),
            'success_rate': deque(maxlen=100),
            'bandwidth': deque(maxlen=100)
        }
        
    def update_strategy(self):
        avg_latency = np.mean(self.metrics['latency'])
        if avg_latency > JITSI_CONFIG['optimization']['max_latency']:
            return {
                'compression_level': min(9, self.current_compression + 1),
                'sampling_rate': self.current_sampling * 0.8
            }
```

## 4. 测试用例补充

```python
# tests/connect/test_jitsi_transport.py
class TestJitsiTransport:
    @pytest.mark.asyncio
    async def test_transport_fallback(self):
        sender = PoseSender(config=test_config)
        
        # 模拟 Jitsi 传输失败
        sender.transport._fail_next = True
        pose_data = create_test_pose()
        
        success = await sender.send_pose_data(pose_data)
        assert success  # 应该通过备份传输成功
        assert sender._active_transport == sender.fallback
        
    @pytest.mark.asyncio
    async def test_performance_degradation(self):
        transport = JitsiTransport(config=test_config)
        
        # 模拟网络延迟
        for _ in range(100):
            transport._stats['latency'].append(0.15)  # 150ms
            
        # 验证是否触发丢帧
        assert transport._should_drop_frame()
```

## 5. 监控指标

### 5.1 关键指标
- 传输延迟
- 丢帧率
- 切换频率
- 压缩率
- 带宽使用

### 5.2 告警阈值
```python
ALERT_THRESHOLDS = {
    'latency': 200,      # ms
    'drop_rate': 0.1,    # 10%
    'switch_rate': 0.05, # 5%
    'compression': 0.3   # 30%
}
```

## 6. 回滚方案

### 6.1 快速回滚
```python
async def rollback():
    # 1. 切换到备份传输
    pose_sender._active_transport = pose_sender.fallback
    
    # 2. 关闭 Jitsi 连接
    await pose_sender.transport.channel.close()
    
    # 3. 恢复配置
    load_backup_config()
```

### 6.2 数据恢复
- 保留最近 100 帧数据
- 记录切换时间点
- 支持重放关键帧

## 7. 部署检查清单

- [ ] Jitsi 服务器配置验证
- [ ] STUN/TURN 服务器测试
- [ ] 网络端口开放确认
- [ ] 性能基准测试
- [ ] 监控系统就绪
- [ ] 备份系统确认
- [ ] 回滚方案演练 
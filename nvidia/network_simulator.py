import time
import random
import threading
import logging
from typing import Dict, Optional, List, Callable

logger = logging.getLogger(__name__)

class NetworkSimulator:
    """网络带宽模拟器，用于模拟不同网络环境下的性能测试"""
    
    NETWORK_PROFILES = {
        "high": {
            "bandwidth_kbps": 500,  # 5Mbps
            "latency_ms": 20,
            "jitter_ms": 5,
            "packet_loss": 0.01,  # 1%
        },
        "medium": {
            "bandwidth_kbps": 10,  # 1Mbps
            "latency_ms": 50,
            "jitter_ms": 15,
            "packet_loss": 0.03,  # 3%
        },
        "low": {
            "bandwidth_kbps": 3,  # 300Kbps
            "latency_ms": 100,
            "jitter_ms": 30,
            "packet_loss": 0.05,  # 5%
        },
        "unstable": {
            "bandwidth_kbps": 8,  # 800Kbps
            "latency_ms": 150,
            "jitter_ms": 80,
            "packet_loss": 0.1,  # 10%
        },
        "mobile": {
            "bandwidth_kbps": 500,  # 500Kbps
            "latency_ms": 80,
            "jitter_ms": 25,
            "packet_loss": 0.04,  # 4%
        }
    }
    
    def __init__(self, profile: str = "high"):
        """初始化网络模拟器
        
        Args:
            profile: 网络配置文件名称，可选 "high", "medium", "low", "unstable", "mobile"
        """
        self.set_profile(profile)
        self.is_running = False
        self.thread = None
        self.lock = threading.RLock()
        self.bandwidth_usage = 0  # 当前带宽使用量 (bytes/sec)
        self.total_bytes_sent = 0
        self.start_time = 0
        self.packet_counter = 0
        self.dropped_packets = 0
        self.callbacks = []  # 带宽变化回调函数列表
        
    def set_profile(self, profile: str) -> bool:
        """设置网络配置文件
        
        Args:
            profile: 网络配置文件名称
            
        Returns:
            是否成功设置
        """
        if profile in self.NETWORK_PROFILES:
            self.profile = self.NETWORK_PROFILES[profile].copy()
            # 添加动态变化
            self.profile["variation"] = 0.2  # 带宽变化范围 (±20%)
            logger.info(f"设置网络配置文件: {profile}")
            return True
        else:
            logger.warning(f"无效的网络配置文件: {profile}")
            return False
            
    def start(self) -> bool:
        """启动网络模拟器"""
        if self.is_running:
            return True
            
        self.is_running = True
        self.start_time = time.time()
        self.total_bytes_sent = 0
        self.packet_counter = 0
        self.dropped_packets = 0
        self.thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.thread.start()
        
        logger.info(f"网络模拟器已启动，带宽: {self.profile['bandwidth_kbps']}kbps, "
                  f"延迟: {self.profile['latency_ms']}ms")
        return True
        
    def stop(self) -> None:
        """停止网络模拟器"""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        logger.info("网络模拟器已停止")
        
    def _simulation_loop(self) -> None:
        """模拟网络波动"""
        while self.is_running:
            try:
                # 模拟带宽变化
                variation = random.uniform(
                    1.0 - self.profile["variation"], 
                    1.0 + self.profile["variation"]
                )
                current_bandwidth = self.profile["bandwidth_kbps"] * variation
                
                # 根据不同时间段调整带宽模拟真实网络环境
                hour = time.localtime().tm_hour
                if 9 <= hour <= 11 or 19 <= hour <= 22:  # 高峰期
                    current_bandwidth *= 0.8  # 降低20%
                
                # 计算当前的带宽使用情况
                elapsed = time.time() - self.start_time
                if elapsed > 0:
                    with self.lock:
                        self.bandwidth_usage = (self.total_bytes_sent * 8 / 1000) / elapsed  # Kbps
                
                # 通知回调
                for callback in self.callbacks:
                    try:
                        callback({
                            'bandwidth_kbps': current_bandwidth,
                            'usage_kbps': self.bandwidth_usage,
                            'packet_loss': self.dropped_packets / max(1, self.packet_counter),
                            'latency_ms': self.profile['latency_ms']
                        })
                    except Exception as e:
                        logger.error(f"执行带宽回调时出错: {e}")
                
                # 每秒更新一次
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"网络模拟循环出错: {e}")
                time.sleep(1.0)
                
    def register_callback(self, callback: Callable) -> None:
        """注册带宽变化回调函数
        
        Args:
            callback: 回调函数，接收一个包含网络状态的字典
        """
        self.callbacks.append(callback)
        
    def unregister_callback(self, callback: Callable) -> None:
        """注销带宽变化回调函数"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def simulate_send(self, data_size: int) -> bool:
        """模拟发送数据包
        
        Args:
            data_size: 数据包大小 (bytes)
            
        Returns:
            是否成功发送
        """
        with self.lock:
            self.packet_counter += 1
            
            # 模拟丢包
            if random.random() < self.profile["packet_loss"]:
                self.dropped_packets += 1
                return False
                
            # 模拟延迟
            jitter = random.uniform(-self.profile["jitter_ms"], self.profile["jitter_ms"])
            latency = self.profile["latency_ms"] + jitter
            time.sleep(max(0, latency / 1000.0))
                
            # 模拟带宽限制
            current_bandwidth = self.profile["bandwidth_kbps"] * 1000 / 8  # 转换为bytes/s
            if self.bandwidth_usage > current_bandwidth:
                # 带宽超限，随机丢弃
                if random.random() < 0.5:
                    self.dropped_packets += 1
                    return False
            
            # 记录发送的字节数
            self.total_bytes_sent += data_size
            return True
            
    def get_status(self) -> Dict:
        """获取网络状态"""
        elapsed = time.time() - self.start_time
        packet_loss = self.dropped_packets / max(1, self.packet_counter) if self.packet_counter > 0 else 0
        
        return {
            'running': self.is_running,
            'profile': next((k for k, v in self.NETWORK_PROFILES.items() 
                          if v['bandwidth_kbps'] == self.profile['bandwidth_kbps']), "custom"),
            'bandwidth_kbps': self.profile['bandwidth_kbps'],
            'current_usage_kbps': self.bandwidth_usage,
            'latency_ms': self.profile['latency_ms'],
            'jitter_ms': self.profile['jitter_ms'],
            'packet_loss': packet_loss,
            'packets_sent': self.packet_counter,
            'packets_dropped': self.dropped_packets,
            'bytes_sent': self.total_bytes_sent,
            'duration_sec': elapsed
        }

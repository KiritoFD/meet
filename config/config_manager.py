import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass
import time
from connect.errors import ConfigurationError

@dataclass
class SystemConfig:
    server: Dict[str, Any]
    socket: Dict[str, Any]
    audio: Dict[str, Any]
    pose: Dict[str, Any]

class ConfigManager:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = None
        self.load_attempts = 0
        self.max_attempts = 3
        self._load_config()
        
    def _load_config(self) -> SystemConfig:
        """加载配置文件"""
        while self.load_attempts < self.max_attempts:
            try:
                if not self.config_path.exists():
                    self._create_default_config()
                    
                with open(self.config_path) as f:
                    config_dict = yaml.safe_load(f)
                    
                if not self._validate_config(config_dict):
                    raise ConfigurationError("Invalid configuration format")
                    
                self.config = SystemConfig(**config_dict)
                return self.config
                
            except Exception as e:
                self.load_attempts += 1
                if self.load_attempts >= self.max_attempts:
                    raise ConfigurationError(f"Failed to load config after {self.max_attempts} attempts: {str(e)}")
                time.sleep(1)  # Wait before retry
                
    def _validate_config(self, config: Dict) -> bool:
        """验证配置格式"""
        required_keys = {'server', 'socket', 'audio', 'pose'}
        return all(key in config for key in required_keys)
        
    def _create_default_config(self):
        """创建默认配置"""
        default_config = {
            'server': {'host': 'localhost', 'port': 5000},
            'socket': {'timeout': 30, 'reconnect': True},
            'audio': {'enabled': True, 'sample_rate': 44100},
            'pose': {'model_complexity': 1, 'min_detection_confidence': 0.5}
        }
        
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f)
        
    def get_config(self) -> SystemConfig:
        """获取系统配置"""
        return self.config
        
    def update_config(self, new_config: Dict[str, Any]):
        """更新配置"""
        # 合并配置
        config_dict = {
            'server': {**self.config.server, **new_config.get('server', {})},
            'socket': {**self.config.socket, **new_config.get('socket', {})},
            'audio': {**self.config.audio, **new_config.get('audio', {})},
            'pose': {**self.config.pose, **new_config.get('pose', {})}
        }
        
        # 保存到文件
        with open(self.config_path, 'w') as f:
            yaml.dump(config_dict, f)
            
        # 更新内存中的配置
        self.config = SystemConfig(**config_dict) 
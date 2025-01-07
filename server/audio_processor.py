import pyaudio
import wave
import numpy as np
import opuslib
from typing import Optional, Tuple, List
from dataclasses import dataclass
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AudioConfig:
    channels: int = 1
    rate: int = 16000
    chunk: int = 1024
    format: int = pyaudio.paFloat32
    opus_bitrate: int = 32000
    opus_frame_size: int = 960
    noise_reduction: bool = True
    auto_gain_control: bool = True

class AudioProcessor:
    def __init__(self, config: Optional[AudioConfig] = None):
        """初始化音频处理器"""
        self.config = config or AudioConfig()
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.encoder = None
        self.decoder = None
        self.is_recording = False
        self.audio_buffer: List[np.ndarray] = []
        self._initialize_opus()
        
    def _initialize_opus(self):
        """初始化Opus编解码器"""
        try:
            self.encoder = opuslib.Encoder(
                self.config.rate,
                self.config.channels,
                opuslib.APPLICATION_VOIP
            )
            self.encoder.bitrate = self.config.opus_bitrate
            
            self.decoder = opuslib.Decoder(
                self.config.rate,
                self.config.channels
            )
            logger.info("Opus编解码器初始化成功")
        except Exception as e:
            logger.error(f"Opus编解码器初始化失败: {e}")
            raise
        
    def start_recording(self) -> bool:
        """开始录音"""
        if self.is_recording:
            return True
            
        try:
            self.stream = self.audio.open(
                format=self.config.format,
                channels=self.config.channels,
                rate=self.config.rate,
                input=True,
                frames_per_buffer=self.config.chunk,
                stream_callback=self._audio_callback
            )
            self.is_recording = True
            logger.info("开始录音")
            return True
        except Exception as e:
            logger.error(f"启动录音失败: {e}")
            return False
            
    def stop_recording(self):
        """停止录音"""
        if self.stream and self.is_recording:
            self.is_recording = False
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            logger.info("停止录音")
            
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """音频回调函数"""
        if status:
            logger.warning(f"音频回调状态: {status}")
            
        # 转换为numpy数组
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # 处理音频数据
        processed_data, _ = self.process_audio(audio_data)
        
        # 添加到缓冲区
        self.audio_buffer.append(processed_data)
        if len(self.audio_buffer) > 10:  # 保持最近10帧
            self.audio_buffer.pop(0)
            
        return (in_data, pyaudio.paContinue)
            
    def read_audio(self) -> Optional[np.ndarray]:
        """读取音频数据"""
        if not self.stream:
            return None
            
        try:
            data = self.stream.read(self.config.chunk)
            return np.frombuffer(data, dtype=np.float32)
        except Exception as e:
            logger.error(f"读取音频失败: {e}")
            return None
            
    def process_audio(self, audio_data: np.ndarray) -> Tuple[np.ndarray, float]:
        """处理音频数据（降噪、音量标准化等）"""
        try:
            # 1. 降噪
            if self.config.noise_reduction:
                # TODO: 实现降噪算法
                # - 频谱减法
                # - 维纳滤波
                pass
                
            # 2. 自动增益控制
            if self.config.auto_gain_control:
                # TODO: 实现自动增益控制
                # - 计算RMS
                # - 动态范围压缩
                pass
                
            # 3. 音量标准化
            max_amplitude = np.max(np.abs(audio_data))
            if max_amplitude > 0:
                normalized_data = audio_data / max_amplitude
            else:
                normalized_data = audio_data
                
            return normalized_data, max_amplitude
            
        except Exception as e:
            logger.error(f"音频处理失败: {e}")
            return audio_data, 1.0
        
    def encode_audio(self, audio_data: np.ndarray) -> bytes:
        """编码音频数据"""
        try:
            # 将float32转换为int16
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # 编码
            encoded = self.encoder.encode(
                audio_int16.tobytes(),
                self.config.opus_frame_size
            )
            
            return encoded
        except Exception as e:
            logger.error(f"音频编码失败: {e}")
            return b""
        
    def decode_audio(self, encoded_data: bytes) -> np.ndarray:
        """解码音频数据"""
        try:
            # 解码
            decoded = self.decoder.decode(
                encoded_data,
                self.config.opus_frame_size
            )
            
            # 转换回float32
            audio_array = np.frombuffer(decoded, dtype=np.int16)
            return audio_array.astype(np.float32) / 32767
        except Exception as e:
            logger.error(f"音频解码失败: {e}")
            return np.zeros(self.config.chunk, dtype=np.float32)
        
    def save_to_file(self, filename: str, audio_data: np.ndarray):
        """保存音频到文件"""
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.config.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.config.format))
                wf.setframerate(self.config.rate)
                
                # 将float32转换为int16
                audio_int16 = (audio_data * 32767).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())
                
            logger.info(f"音频保存成功: {filename}")
        except Exception as e:
            logger.error(f"音频保存失败: {e}")
            
    def get_audio_devices(self) -> List[dict]:
        """获取可用的音频设备列表"""
        devices = []
        try:
            for i in range(self.audio.get_device_count()):
                device_info = self.audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:  # 只获取输入设备
                    devices.append({
                        'index': i,
                        'name': device_info['name'],
                        'channels': device_info['maxInputChannels'],
                        'sample_rate': int(device_info['defaultSampleRate'])
                    })
        except Exception as e:
            logger.error(f"获取音频设备列表失败: {e}")
            
        return devices
        
    def set_input_device(self, device_index: int) -> bool:
        """设置输入设备"""
        try:
            device_info = self.audio.get_device_info_by_index(device_index)
            if device_info['maxInputChannels'] > 0:
                # 如果正在录音，先停止
                if self.is_recording:
                    self.stop_recording()
                    
                # 更新配置
                self.config.channels = min(
                    device_info['maxInputChannels'],
                    self.config.channels
                )
                return True
        except Exception as e:
            logger.error(f"设置输入设备失败: {e}")
            
        return False
        
    def __del__(self):
        """清理资源"""
        self.stop_recording()
        if self.audio:
            self.audio.terminate() 
import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        self.sample_rate = 44100
        self.channels = 1
        self.is_recording = False
        self.frames = []
        self.recording_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'audio', 'recordings')
        
        # 确保录音目录存在
        if not os.path.exists(self.recording_path):
            os.makedirs(self.recording_path)

    def start_recording(self):
        """开始录音"""
        if self.is_recording:
            return False
            
        self.frames = []  # 清空之前的录音
        self.is_recording = True
        
        def callback(indata, frames, time, status):
            if status:
                logger.warning(f"录音回调状态: {status}")
            if self.is_recording:
                self.frames.append(indata.copy())
                
        try:
            self.stream = sd.InputStream(
                channels=self.channels,
                samplerate=self.sample_rate,
                callback=callback
            )
            self.stream.start()
            logger.info("开始录音")
            return True
        except Exception as e:
            logger.error(f"启动录音失败: {e}")
            self.is_recording = False
            return False

    def stop_recording(self):
        """停止录音"""
        if not self.is_recording:
            return False, "没有正在进行的录音"

        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        try:
            return self.save_audio()
        except Exception as e:
            logger.error(f"保存录音失败: {e}")
            return False, str(e)

    def save_audio(self):
        """保存录音文件"""
        if not self.frames:
            return False, "没有录音数据"
            
        try:
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.wav"
            filepath = os.path.join(self.recording_path, filename)
            
            # 合并所有帧
            audio_data = np.concatenate(self.frames, axis=0)
            
            # 保存文件
            sf.write(filepath, audio_data, self.sample_rate)
            logger.info(f"录音已保存到: {filepath}")
            
            # 清空帧数据
            self.frames = []
            
            return True, filepath
        except Exception as e:
            logger.error(f"保存录音文件失败: {e}")
            return False, str(e)

    def get_status(self):
        """获取录音状态"""
        return {
            "is_recording": self.is_recording,
            "sample_rate": self.sample_rate,
            "channels": self.channels
        }
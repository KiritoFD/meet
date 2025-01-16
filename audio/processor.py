import sounddevice as sd
import numpy as np
import logging
from flask_socketio import SocketIO

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, socketio=None):
        self.sample_rate = 44100
        self.channels = 1
        self.is_recording = False
        self.frames = []
        self.socketio = socketio
        self.stream = None

    def set_socketio(self, socketio):
        """设置socketio实例"""
        self.socketio = socketio

    def start_recording(self):
        """开始录音"""
        if self.is_recording:
            return False
            
        def callback(indata, frames, time, status):
            if status:
                logger.warning(f"录音回调状态: {status}")
            if self.is_recording:
                self.frames.append(indata.copy())
                # 计算音量并发送
                volume_norm = float(np.linalg.norm(indata) * 10)
                if self.socketio:
                    try:
                        self.socketio.emit('volume_update', {'volume': volume_norm})
                    except Exception as e:
                        logger.error(f"发送音量数据失败: {e}")

        try:
            self.stream = sd.InputStream(
                channels=self.channels,
                samplerate=self.sample_rate,
                callback=callback
            )
            self.stream.start()
            self.is_recording = True
            logger.info("开始录音")
            return True
        except Exception as e:
            logger.error(f"启动录音失败: {e}")
            self.is_recording = False
            return False

    def stop_recording(self):
        """停止录音"""
        if not self.is_recording:
            return False

        try:
            self.is_recording = False
            if self.stream:
                self.stream.stop()
                self.stream.close()
            self.stream = None
            self.frames = []
            logger.info("停止录音")
            return True
        except Exception as e:
            logger.error(f"停止录音失败: {e}")
            return False

    def get_status(self):
        """获取录音状态"""
        return {
            'is_recording': self.is_recording,
            'sample_rate': self.sample_rate,
            'buffer_size': len(self.frames) if self.frames else 0
        }
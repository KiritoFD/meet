import logging
import numpy as np
import time
import threading
from typing import List, Optional

# Set up logging
logger = logging.getLogger(__name__)

# Try to import sounddevice, but provide a fallback if it's not available
try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
    logger.info("Audio processing is available (sounddevice loaded)")
except OSError as e:
    AUDIO_AVAILABLE = False
    logger.warning(f"PortAudio library not found: {str(e)}")
    logger.warning("Audio processing will be disabled. To enable audio, install PortAudio.")
    logger.info("On Windows: pip install pipwin && pipwin install pyaudio")
    logger.info("On Linux: sudo apt-get install portaudio19-dev && pip install pyaudio")
    logger.info("On macOS: brew install portaudio && pip install pyaudio")

class AudioProcessor:
    """Audio processing class with fallback when PortAudio is not available"""
    
    def __init__(self, sample_rate=44100, channels=1, chunk_size=1024):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.frames = []
        self.is_recording = False
        self.stream = None
        self.socketio = None
        self.audio_available = AUDIO_AVAILABLE
        self.thread = None
        
        if not self.audio_available:
            logger.warning("AudioProcessor initialized but audio capture is not available")
    
    def set_socketio(self, socketio):
        """Set the SocketIO instance for emitting audio data"""
        self.socketio = socketio
        
    def audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio stream"""
        if not self.audio_available:
            return
            
        if self.is_recording:
            self.frames.append(indata.copy())
            if self.socketio:
                audio_data = indata.tobytes()
                self.socketio.emit('audio_chunk', {'data': audio_data})
    
    def start_recording(self) -> bool:
        """Start audio recording"""
        if not self.audio_available:
            logger.warning("Cannot start recording: Audio is not available")
            return False
            
        if self.is_recording:
            logger.warning("Recording is already in progress")
            return True
            
        try:
            self.frames = []
            self.is_recording = True
            
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size
            )
            self.stream.start()
            logger.info("Audio recording started")
            return True
        except Exception as e:
            logger.error(f"Failed to start audio recording: {str(e)}")
            self.is_recording = False
            return False
    
    def stop_recording(self) -> bool:
        """Stop audio recording"""
        if not self.audio_available:
            return False
            
        if not self.is_recording:
            logger.warning("No recording in progress")
            return False
            
        try:
            self.is_recording = False
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            logger.info("Audio recording stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop audio recording: {str(e)}")
            return False
    
    def get_audio_data(self) -> Optional[np.ndarray]:
        """Get the recorded audio data as a numpy array"""
        if not self.audio_available or not self.frames:
            return None
            
        try:
            return np.concatenate(self.frames, axis=0)
        except Exception as e:
            logger.error(f"Failed to process audio data: {str(e)}")
            return None
    
    def process_audio(self):
        """Process audio data (placeholder for audio processing logic)"""
        if not self.audio_available:
            return None
            
        audio_data = self.get_audio_data()
        # Add your audio processing logic here
        return audio_data
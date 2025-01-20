import pyaudio

AUDIO_CONFIG = {
    'format': pyaudio.paInt16,
    'channels': 1,
    'rate': 44100,
    'chunk': 1024,
    'record_seconds': 5
} 
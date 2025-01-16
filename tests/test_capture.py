import pytest
from src.core.capture import CaptureManager

def test_capture_manager_init():
    config = {
        "device": 0,
        "width": 1280,
        "height": 720,
        "fps": 30
    }
    manager = CaptureManager(config)
    assert manager.device == 0
    assert manager.frame_size == (1280, 720) 
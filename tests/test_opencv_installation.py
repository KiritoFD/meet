import cv2
import pytest

def test_opencv_import():
    with pytest.raises(ImportError):
        cv2.__import__('cv2')
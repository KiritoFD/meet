import os
import sys
from pathlib import Path
import pytest
import logging
import numpy as np
from unittest.mock import Mock, AsyncMock
from typing import Dict, Optional

# Get project root and add to Python path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure output directory
output_dir = os.path.join(project_root, 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(output_dir, 'test.log')),
        logging.StreamHandler()
    ]
)

# Global cv2 instance to prevent recursion
_cv2 = None

def _cleanup_cv2():
    """Clean up cv2 related imports and paths"""
    # Remove cv2 from sys.modules to prevent recursion
    cv2_related = [name for name in sys.modules if 'cv2' in name]
    for name in cv2_related:
        del sys.modules[name]
    # Clean up Python path
    sys.path = [p for p in sys.path if 'cv2' not in p]

@pytest.fixture(scope='session')
def cv2():
    """Provide cv2 module with lazy loading and recursion prevention"""
    global _cv2
    if _cv2 is None:
        try:
            _cleanup_cv2()
            import cv2 as cv2_import
            _cv2 = cv2_import
        except ImportError:
            pytest.skip("OpenCV (cv2) not available")
    return _cv2

@pytest.fixture(autouse=True)
def _prevent_cv2_recursion():
    """Automatically clean up cv2 imports before each test"""
    _cleanup_cv2()
    yield
    _cleanup_cv2()

@pytest.fixture(scope='session')
def jwt():
    """Provide JWT module with lazy loading"""
    try:
        import jwt
        return jwt
    except ImportError:
        pytest.skip("PyJWT not available")

@pytest.fixture(scope='session')
def test_frame(cv2):
    """Create test image frame"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(frame, (320, 240), 50, (255, 255, 255), -1)
    cv2.rectangle(frame, (100, 100), (200, 200), (128, 128, 128), -1)
    return frame

@pytest.fixture(scope='session')
def test_pose_data():
    """Create test pose data"""
    return {
        'landmarks': [
            {'x': 0.5, 'y': 0.5, 'z': 0.0, 'visibility': 1.0}
            for _ in range(33)
        ]
    }

@pytest.fixture(scope='session')
def mock_camera_manager():
    """Create mock camera manager"""
    manager = Mock()
    manager.is_running = True
    manager.read_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    return manager

@pytest.fixture(scope='session')
def mock_pose():
    """Create mock pose detector"""
    pose = Mock()
    pose.process.return_value = Mock(
        pose_landmarks=Mock(
            landmark=[Mock(x=0.5, y=0.5, z=0.0, visibility=0.9) for _ in range(33)]
        )
    )
    return pose

@pytest.fixture(scope='session')
def mock_face_mesh():
    """Create mock face mesh detector"""
    face_mesh = Mock()
    face_mesh.process.return_value = Mock(multi_face_landmarks=[Mock()])
    return face_mesh

@pytest.fixture(scope='session')
def config():
    """Provide test configuration"""
    return {
        'sender': {
            'queue_size': 1000,
            'batch_size': 10,
            'send_interval': 0.1
        },
        'processor': {
            'compression_level': 6,
            'max_pose_size': 1024 * 1024,
            'batch_size': 10
        },
        'camera': {
            'width': 640,
            'height': 480,
            'fps': 30
        },
        'pose': {
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5
        }
    }

@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup after each test"""
    yield
    # Add cleanup code here if needed
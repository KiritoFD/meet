import cv2
import numpy as np
import base64
import logging

logger = logging.getLogger(__name__)

def encode_image(image: np.ndarray) -> str:
    """将图像编码为base64字符串"""
    try:
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        logger.error(f"图像编码失败: {e}")
        return None

def decode_image(image_str: str) -> np.ndarray:
    """从base64字符串解码图像"""
    try:
        nparr = np.frombuffer(base64.b64decode(image_str), np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"图像解码失败: {e}")
        return None 
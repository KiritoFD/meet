import json
import zlib
import base64
import logging

logger = logging.getLogger(__name__)

def compress_data(data: dict) -> bytes:
    """压缩数据"""
    try:
        json_str = json.dumps(data)
        return zlib.compress(json_str.encode())
    except Exception as e:
        logger.error(f"压缩数据失败: {e}")
        return None

def decompress_data(compressed_data: bytes) -> dict:
    """解压缩数据"""
    try:
        json_str = zlib.decompress(compressed_data).decode()
        return json.loads(json_str)
    except Exception as e:
        logger.error(f"解压缩数据失败: {e}")
        return None 
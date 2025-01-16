from dataclasses import dataclass
from typing import List, Dict, Optional, Union
import json
import zlib
import time
import numpy as np

@dataclass
class MediaData:
    pose_landmarks: Optional[List[Dict[str, float]]] = None
    face_landmarks: Optional[List[Dict[str, float]]] = None
    hand_landmarks: Optional[List[Dict[str, float]]] = None
    audio_data: Optional[bytes] = None
    timestamp: float = 0.0
    frame_id: int = 0
    data_type: str = "pose"  # "pose" or "audio"

class MediaProtocol:
    def __init__(self, compression_level: int = 6):
        self.compression_level = compression_level

    def encode(self, data: MediaData) -> bytes:
        try:
            # Convert to dictionary
            data_dict = {
                "data_type": data.data_type,
                "timestamp": data.timestamp,
                "frame_id": data.frame_id
            }

            if data.data_type == "pose":
                data_dict.update({
                    "pose_landmarks": data.pose_landmarks,
                    "face_landmarks": data.face_landmarks,
                    "hand_landmarks": data.hand_landmarks,
                })
            elif data.data_type == "audio":
                # Convert audio bytes to base64 for JSON serialization
                data_dict["audio_data"] = data.audio_data.hex() if data.audio_data else None

            # Convert to JSON string
            json_data = json.dumps(data_dict)
            
            # Compress
            compressed = zlib.compress(json_data.encode(), self.compression_level)
            return compressed

        except Exception as e:
            raise ValueError(f"Encoding error: {str(e)}")

    def decode(self, data: bytes) -> MediaData:
        try:
            # Decompress
            decompressed = zlib.decompress(data)
            
            # Parse JSON
            data_dict = json.loads(decompressed.decode())
            
            # Convert hex back to bytes for audio data if present
            if data_dict.get("data_type") == "audio" and data_dict.get("audio_data"):
                data_dict["audio_data"] = bytes.fromhex(data_dict["audio_data"])

            # Create MediaData object
            return MediaData(**data_dict)

        except Exception as e:
            raise ValueError(f"Decoding error: {str(e)}")

    def validate(self, data: MediaData) -> bool:
        if data.timestamp <= 0:
            return False
        if data.frame_id < 0:
            return False

        if data.data_type == "pose":
            return data.pose_landmarks is not None
        elif data.data_type == "audio":
            return data.audio_data is not None
        
        return False 
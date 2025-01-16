import time
import logging
from collections import deque
from typing import Dict, Optional
from .socket_manager import SocketManager
from .pose_protocol import MediaData, MediaProtocol

class MediaSender:
    def __init__(self, socket: SocketManager, protocol: MediaProtocol):
        self.socket = socket
        self.protocol = protocol
        self.logger = logging.getLogger(__name__)
        self.frame_count = 0
        self.start_time = time.time()
        self.frame_times = deque(maxlen=100)
        self.stats = {
            "fps": 0.0,
            "latency": 0.0,
            "success_rate": 100.0,
            "failed_frames": 0
        }

    async def send_pose_frame(self, 
                            pose_results,
                            face_results=None, 
                            hands_results=None) -> bool:
        try:
            media_data = MediaData(
                pose_landmarks=pose_results,
                face_landmarks=face_results,
                hand_landmarks=hands_results,
                timestamp=time.time(),
                frame_id=self.frame_count,
                data_type="pose"
            )
            return await self._send_data(media_data)
        except Exception as e:
            self.logger.error(f"Error sending pose frame: {str(e)}")
            return False

    async def send_audio_frame(self, audio_data: bytes) -> bool:
        try:
            media_data = MediaData(
                audio_data=audio_data,
                timestamp=time.time(),
                frame_id=self.frame_count,
                data_type="audio"
            )
            return await self._send_data(media_data)
        except Exception as e:
            self.logger.error(f"Error sending audio frame: {str(e)}")
            return False

    async def _send_data(self, media_data: MediaData) -> bool:
        try:
            start_time = time.time()

            # Validate data
            if not self.protocol.validate(media_data):
                self.logger.error(f"Invalid {media_data.data_type} data")
                return False

            # Encode data
            encoded_data = self.protocol.encode(media_data)

            # Send data
            event_type = f"{media_data.data_type}_frame"
            success = await self.socket.emit(event_type, encoded_data)

            # Update statistics
            self._update_stats(success, start_time)
            
            return success

        except Exception as e:
            self.logger.error(f"Error sending data: {str(e)}")
            self.stats["failed_frames"] += 1
            return False

    def get_stats(self) -> Dict[str, float]:
        return self.stats.copy()

    @property
    def fps(self) -> float:
        return self.stats["fps"]

    def _update_stats(self, success: bool, start_time: float):
        self.frame_count += 1
        end_time = time.time()
        
        # Update FPS
        self.frame_times.append(end_time - start_time)
        if len(self.frame_times) >= 2:
            self.stats["fps"] = 1.0 / (sum(self.frame_times) / len(self.frame_times))

        # Update latency
        self.stats["latency"] = (end_time - start_time) * 1000  # Convert to ms

        # Update success rate
        if not success:
            self.stats["failed_frames"] += 1
        self.stats["success_rate"] = ((self.frame_count - self.stats["failed_frames"]) 
                                    / self.frame_count * 100) 
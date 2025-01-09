import mediapipe as mp
from utils.logger import logger
from config.settings import MEDIAPIPE_CONFIG

class MediaPipeManager:
    def __init__(self):
        self.initialized = False
        try:
            self.pose = mp.solutions.pose.Pose(**MEDIAPIPE_CONFIG['pose'])
            self.hands = mp.solutions.hands.Hands(**MEDIAPIPE_CONFIG['hands'])
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(**MEDIAPIPE_CONFIG['face_mesh'])
            self.initialized = True
            logger.info("MediaPipe 初始化成功")
        except Exception as e:
            logger.error(f"MediaPipe 初始化失败: {e}")
            self.initialized = False

    def process_frame(self, frame_rgb):
        """处理单个帧的所有检测"""
        if not self.initialized:
            logger.error("MediaPipe 未初始化")
            return None
        try:
            if frame_rgb is None:
                logger.error("输入帧为空")
                return None
            
            h, w = frame_rgb.shape[:2]
            logger.debug(f"处理帧大小: {w}x{h}")
            
            # 确保每个处理都返回结果
            pose_results = self.pose.process(frame_rgb)
            hands_results = self.hands.process(frame_rgb)
            face_results = self.face_mesh.process(frame_rgb)
            
            logger.debug(f"处理结果: pose={pose_results is not None}, " +
                        f"hands={hands_results is not None}, " +
                        f"face={face_results is not None}")
            
            results = {
                'pose': pose_results,
                'hands': hands_results,
                'face_mesh': face_results
            }
            
            return results
        except Exception as e:
            logger.error(f"处理帧失败: {e}")
            return None

    def cleanup(self):
        try:
            logger.debug("开始清理 MediaPipe 资源...")
            if hasattr(self, 'pose'):
                self.pose.close()
            if hasattr(self, 'hands'):
                self.hands.close()
            if hasattr(self, 'face_mesh'):
                self.face_mesh.close()
            self.initialized = False
            logger.info("MediaPipe 资源清理完成")
        except Exception as e:
            logger.error(f"MediaPipe 清理失败: {e}") 
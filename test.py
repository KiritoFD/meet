import cv2
import mediapipe as mp
import numpy as np
from pose.pose_binding import PoseBinding, BindingConfig 
from pose.pose_deformer import PoseDeformer
from pose.pose_data import PoseData
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_pose_detector():
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )
    return pose, mp_drawing

def process_frame(frame, pose_detector):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(rgb_frame)
    
    if not results.pose_landmarks:
        logger.debug("未检测到姿态关键点")
        return None, results
        
    landmarks = []
    for landmark in results.pose_landmarks.landmark:
        landmarks.append({
            'x': landmark.x,
            'y': landmark.y, 
            'z': landmark.z,
            'visibility': landmark.visibility
        })
    
    return PoseData(landmarks=landmarks), results

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("无法打开摄像头")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    pose_detector, mp_drawing = init_pose_detector()
    pose_binding = PoseBinding()
    pose_deformer = PoseDeformer()
    
    reference_frame = None
    reference_pose = None
    regions = None
    
    logger.info("系统初始化完成，开始处理...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("无法读取摄像头画面")
                break
                
            pose_data, results = process_frame(frame, pose_detector)
            
            # 显示姿态检测结果
            if results.pose_landmarks:
                output_frame = frame.copy()
                mp_drawing.draw_landmarks(
                    output_frame,
                    results.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS
                )
                cv2.imshow('Pose Detection', output_frame)
            
            if pose_data is None:
                continue
                
            if reference_frame is None:
                logger.info("捕获参考帧...")
                reference_frame = frame.copy()
                reference_pose = pose_data
                regions = pose_binding.create_binding(reference_frame, reference_pose)
                logger.info(f"创建了 {len(regions)} 个绑定区域")
                continue
                
            try:
                updated_regions = pose_binding.update_binding(regions, pose_data)
                
                if updated_regions:
                    deformed_frame = pose_deformer.deform(
                        reference_frame,
                        reference_pose,
                        frame,
                        pose_data,
                        updated_regions
                    )
                    
                    if deformed_frame is not None:
                        cv2.imshow('Deformed Result', deformed_frame)
                    cv2.imshow('Original', frame)
                    
            except Exception as e:
                logger.error(f"处理失败: {str(e)}")
                continue
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("用户退出程序")
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose_detector.close()
        logger.info("程序已结束")

if __name__ == '__main__':
    main()
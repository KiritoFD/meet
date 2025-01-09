from flask import render_template, Response
from utils.logger import logger
from core.visualizer import Visualizer
import cv2
import time
import mediapipe as mp

def init_main_routes(app, camera_manager, mp_manager):
    visualizer = Visualizer()
    last_frame_time = 0
    
    @app.route('/')
    def index():
        """主页"""
        return render_template('display.html')

    @app.route('/receiver')
    def receiver():
        """接收端页面"""
        return render_template('receiver.html')

    @app.route('/video_feed')
    def video_feed():
        """视频流"""
        def generate_frames():
            nonlocal last_frame_time
            
            while True:
                frame = camera_manager.read_frame()
                logger.debug("读取帧状态: %s", "成功" if frame is not None else "失败")
                if frame is None:
                    time.sleep(0.1)  # 避免CPU过度使用
                    continue
                
                # 处理帧
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = mp_manager.process_frame(frame_rgb)
                logger.debug("MediaPipe处理结果: %s", results is not None)
                
                if results is None:
                    logger.error("MediaPipe处理失败")
                    continue
                
                # 直接在原始帧上绘制
                if results['pose'] and results['pose'].pose_landmarks:
                    logger.debug("开始绘制姿态")
                    h, w = frame.shape[:2]
                    landmarks = results['pose'].pose_landmarks
                    
                    # 绘制姿态关键点
                    for idx, landmark in enumerate(landmarks.landmark):
                        if landmark.visibility > 0.5:
                            x = int(landmark.x * w)
                            y = int(landmark.y * h)
                            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                            cv2.putText(frame, str(idx), (x+5, y+5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # 使用 MediaPipe 提供的绘制工具
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, 
                        landmarks,
                        mp.solutions.pose.POSE_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                    )
                    logger.debug("姿态绘制完成")

                # 绘制手部检测结果
                if results['hands'] and results['hands'].multi_hand_landmarks:
                    logger.debug("开始绘制手部")
                    for hand_landmarks in results['hands'].multi_hand_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp.solutions.hands.HAND_CONNECTIONS,
                            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                            mp.solutions.drawing_styles.get_default_hand_connections_style()
                        )
                    logger.debug("手部绘制完成")

                # 绘制面部网格
                if results['face_mesh'] and results['face_mesh'].multi_face_landmarks:
                    logger.debug("开始绘制面部")
                    for face_landmarks in results['face_mesh'].multi_face_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame,
                            face_landmarks,
                            mp.solutions.face_mesh.FACEMESH_TESSELATION,
                            mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
                        )
                    logger.debug("面部绘制完成")
                
                # 计算并显示FPS
                current_time = time.time()
                fps = 1 / (current_time - last_frame_time) if last_frame_time > 0 else 0
                last_frame_time = current_time
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                # 转换帧格式
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                
                frame_data = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        
        return Response(
            generate_frames(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        ) 
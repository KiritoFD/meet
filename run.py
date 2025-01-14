import os
import sys
from flask import Flask, Response, render_template, jsonify, send_from_directory, request
import cv2
import mediapipe as mp
import numpy as np
import logging
import time
from flask_socketio import SocketIO, emit
from connect.pose_sender import PoseSender
from connect.socket_manager import SocketManager
from camera.manager import CameraManager
from pose.drawer import PoseDrawer
import asyncio
from config import settings

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 获取项目根目录的绝对路径
project_root = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(project_root, 'frontend', 'pages')
static_dir = os.path.join(project_root, 'frontend', 'static')

app = Flask(__name__, 
           template_folder=template_dir,
           static_folder=static_dir,
           static_url_path='/static')

# MediaPipe 初始化
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=True,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 全局变量
camera_manager = CameraManager()
pose_drawer = PoseDrawer()
current_frame = None
current_pose = None

# 初始化
socketio = SocketIO(app, cors_allowed_origins="*")
socket_manager = SocketManager(socketio)
pose_sender = PoseSender(socketio, socket_manager.room_manager)

@app.route('/')
def index():
    return render_template('display.html')

@app.route('/start_capture', methods=['POST'])
def start_capture():
    global camera_manager
    start_time = time.time()
    logger.info("收到启动摄像头请求")
    try:
        logger.info("开始初始化摄像头...")
        success = camera_manager.start()
        init_time = time.time() - start_time
        logger.info(f"摄像头初始化{'成功' if success else '失败'}")
        logger.info(f"总处理时间: {init_time:.2f}秒")
        if not success:
            raise Exception("无法打开摄像头")
        return jsonify({"status": "success"}), 200
    except Exception as e:
        logger.error(f"启动摄像头失败: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/stop_capture', methods=['POST'])
def stop_capture():
    try:
        camera_manager.stop()
        return jsonify({"status": "success"}), 200
    except Exception as e:
        logger.error(f"关闭摄像头失败: {e}")
        return jsonify({"error": str(e)}), 500

def generate_frames():
    global camera_manager, current_frame, current_pose
    
    while True:
        try:
            if not camera_manager.is_running:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                continue

            success, frame = camera_manager.read()
            if not success or frame is None:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 处理姿势
            pose_results = pose.process(frame_rgb)
            
            # 处理手部
            hands_results = hands.process(frame_rgb)

            # 处理面部
            face_results = face_mesh.process(frame_rgb)

            # 发送姿态数据到房间
            pose_sender.send_pose_data(
                room="default_room",  # 或从session获取当前房间
                pose_results=pose_results,
                face_results=face_results,
                hands_results=hands_results,
                timestamp=time.time()
            )

            # 使用绘制器绘制关键点
            pose_drawer.draw_pose(frame, pose_results)
            pose_drawer.draw_hands(frame, hands_results)
            pose_drawer.draw_face(frame, face_results)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            logger.error(f"处理帧时出错: {str(e)}")
            continue

@app.route('/video_feed')
def video_feed():
    try:
        return Response(generate_frames(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logger.error(f"视频流出错: {str(e)}")
        return "视频流错误", 500

@app.route('/pose')
def get_pose():
    if current_pose is None:
        return jsonify([])
    return jsonify(current_pose)

def restart_camera():
    global camera
    if camera is not None:
        camera.release()
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise Exception("无法重新打开摄像头")
    return camera.isOpened()

@app.route('/restart_camera', methods=['POST'])
def handle_restart_camera():
    try:
        success = restart_camera()
        if success:
            return jsonify({"message": "摄像头已重启", "status": "success"})
        else:
            return jsonify({"error": "重启摄像头失败", "status": "error"}), 500
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500

# 添加摄像头状态检查路由
@app.route('/camera_status')
def camera_status():
    global camera_manager
    is_running = camera_manager.camera is not None and camera_manager.camera.isOpened()
    return jsonify({
        "isRunning": is_running,
        "status": "running" if is_running else "stopped"
    })

@app.route('/log', methods=['POST'])
def handle_frontend_log():
    try:
        log_data = request.json
        level = log_data.get('level', 'info').upper()
        message = log_data.get('message', '')
        data = log_data.get('data')
        source = log_data.get('source', 'unknown')
        timestamp = log_data.get('timestamp')
        
        log_message = f"[{source}] {message}"
        if data:
            log_message += f" - {data}"
        
        if level == 'ERROR':
            logger.error(log_message)
        elif level == 'WARNING':
            logger.warning(log_message)
        else:
            logger.info(log_message)
            
        return jsonify({"status": "success"}), 200
    except Exception as e:
        logger.error(f"处理前端日志失败: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # 确保必要的目录存在
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print(f"服务器启动在 http://localhost:5000")
    print(f"模板目录: {template_dir}")
    print(f"静态文件目录: {static_dir}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
import os
import sys
from flask import Flask, Response, render_template, jsonify, send_from_directory, request
from werkzeug.utils import secure_filename
import cv2
import mediapipe as mp
import numpy as np
import logging
import time
from flask_socketio import SocketIO, emit
from camera.manager import CameraManager
from pose.drawer import PoseDrawer  # 确保从正确的路径导入
from connect.pose_sender import PoseSender
from connect.socket_manager import SocketManager
from config import settings
from config.settings import CAMERA_CONFIG
from audio.processor import AudioProcessor

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

# 初始化音频处理器
audio_processor = AudioProcessor()

# 定义上传文件夹路径
UPLOAD_FOLDER = os.path.join(project_root, 'uploads')

# 初始化 Socket.IO
socketio = SocketIO(app, cors_allowed_origins="*")
socket_manager = SocketManager(socketio, audio_processor)
pose_sender = PoseSender(socketio, socket_manager)

# MediaPipe 初始化
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# 初始化 MediaPipe 模型
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

# 初始化处理器
audio_processor = AudioProcessor()
audio_processor.set_socketio(socketio)

def check_camera_settings(cap):
    """检查摄像头实际参数"""
    logger.info("摄像头当前参数:")
    params = {
        cv2.CAP_PROP_EXPOSURE: "曝光值",
        cv2.CAP_PROP_BRIGHTNESS: "亮度",
        cv2.CAP_PROP_CONTRAST: "对比度",
        cv2.CAP_PROP_GAIN: "增益"
    }
    
    for param, name in params.items():
        value = cap.get(param)
        logger.info(f"{name}: {value}")

# 在摄像头初始化后添加:
cap = cv2.VideoCapture(CAMERA_CONFIG['device_id'], CAMERA_CONFIG['api_preference'])
for param, value in CAMERA_CONFIG['params'].items():
    cap.set(param, value)

# 检查设置是否生效
check_camera_settings(cap)

@app.route('/')
def index():
    return render_template('display.html')

@app.route('/start_capture', methods=['POST'])
def start_capture():
    success = camera_manager.start()
    return jsonify({'success': success})

@app.route('/stop_capture', methods=['POST'])
def stop_capture():
    success = camera_manager.stop()
    return jsonify({'success': success})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_audio', methods=['POST'])
def start_audio():
    success = audio_processor.start_recording()
    return jsonify({'success': success})

@app.route('/stop_audio', methods=['POST'])
def stop_audio():
    success = audio_processor.stop_recording()
    return jsonify({'success': success})

@app.route('/check_stream_status')
def check_stream_status():
    try:
        status = {
            'video': {
                'is_streaming': camera_manager.is_running,
                'fps': camera_manager.current_fps
            },
            'audio': {
                'is_recording': audio_processor.is_recording,
                'sample_rate': audio_processor.sample_rate,
                'buffer_size': len(audio_processor.frames) if hasattr(audio_processor, 'frames') else 0
            }
        }
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"获取流状态失败: {str(e)}")
        return jsonify({'error': str(e)}), 500

def generate_frames():
    while True:
        try:
            if not camera_manager.is_running:
                # 创建空白帧
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Camera Not Running", (180, 240),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                success, frame = camera_manager.read()
                if not success or frame is None:
                    continue

                # 转换颜色空间用于MediaPipe处理
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 处理各个模型
                pose_results = pose.process(frame_rgb)
                hands_results = hands.process(frame_rgb)
                face_results = face_mesh.process(frame_rgb)
                
                # 使用PoseDrawer绘制
                frame = pose_drawer.draw_frame(
                    frame,
                    pose_results,
                    hands_results,
                    face_results
                )
                
                # 发送姿态数据
                pose_sender.send_pose_data(
                    room="default_room",
                    pose_results=pose_results,
                    face_results=face_results,
                    hands_results=hands_results,
                    timestamp=time.time()
                )

                # 添加FPS显示
                cv2.putText(frame, f"FPS: {camera_manager.current_fps}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                          1, (0, 255, 0), 2)

            # 编码并发送帧
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            logger.error(f"处理帧时出错: {str(e)}")
            time.sleep(0.1)  # 出错时短暂暂停
            continue

@app.route('/camera_status')
def camera_status():
    try:
        status = {
            "isRunning": camera_manager.is_running,
            "fps": camera_manager.current_fps,
            "status": "running" if camera_manager.is_running else "stopped"
        }
        return jsonify(status)
    except Exception as e:
        logger.error(f"获取摄像头状态失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@socketio.on('connect')
def handle_connect():
    logger.info('客户端已连接')

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('客户端已断开连接')

@app.route('/api/upload_audio', methods=['POST'])
def upload_audio():
    """上传音频文件"""
    try:
        if 'audio' not in request.files:
            return jsonify({
                'status': 'error',
                'message': '没有上传文件'
            }), 400
            
        file = request.files['audio']
        if file.filename == '':
            return jsonify({
                'status': 'error', 
                'message': '未选择文件'
            }), 400
            
        # 确保上传目录存在
        audio_dir = os.path.join(UPLOAD_FOLDER, 'audio')
        os.makedirs(audio_dir, exist_ok=True)
        
        # 保存文件
        filename = secure_filename(file.filename)
        file_path = os.path.join(audio_dir, filename)
        file.save(file_path)
        
        return jsonify({
            'status': 'success',
            'message': '音频上传成功',
            'audio_url': os.path.join('/uploads/audio', filename)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/audio/<filename>')
def stream_audio(filename):
    """流式传输音频文件"""
    def generate():
        audio_path = os.path.join(UPLOAD_FOLDER, 'audio', filename)
        with open(audio_path, 'rb') as audio_file:
            data = audio_file.read(1024)
            while data:
                yield data
                data = audio_file.read(1024)
                
    return Response(generate(), mimetype='audio/mpeg')

if __name__ == '__main__':
    # 确保必要的目录存在
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    logger.info(f"服务器启动在 http://localhost:5000")
    logger.info(f"模板目录: {template_dir}")
    logger.info(f"静态文件目录: {static_dir}")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
from flask import Flask, Response, render_template, jsonify, request, session
from flask_socketio import SocketIO, emit, join_room, leave_room
import cv2
import mediapipe as mp
import numpy as np
import logging
import time
import json
import zlib
import base64
import uuid
import sys
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# 应用配置
app = Flask(__name__, 
    static_folder='static',
    static_url_path='/static'
)
app.config['SECRET_KEY'] = 'your_secret_key_here'
socketio = SocketIO(app, cors_allowed_origins="*")

# 全局变量
camera = None
initial_frame = None
last_pose_update = 0
current_room = None

# 配置
CAMERA_CONFIG = {
    'index': 0,
    'width': 640,
    'height': 480,
    'fps': 30
}

POSE_CONFIG = {
    'update_interval': 100,  # ms
    'smoothing_factor': 0.5
}

# 房间管理
class Room:
    def __init__(self, room_id):
        self.id = room_id
        self.sender = None
        self.receiver = None
        self.initial_frame = None
        self.ready = False

rooms = {}

# 添加 MediaPipe 管理器
class MediaPipeManager:
    def __init__(self):
        self.pose = None
        self.initialized = False
        
    def get_pose(self):
        if not self.initialized:
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=0,  # 降低模型复杂度，加快处理速度
                enable_segmentation=False,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.initialized = True
        return self.pose

mp_manager = MediaPipeManager()

# 添加视频相关函数
def initialize_camera():
    """优化的摄像头初始化"""
    global camera
    try:
        if camera is not None:
            camera.release()
        
        logger.info("正在初始化摄像头...")
        camera = cv2.VideoCapture(CAMERA_CONFIG['index'])
        
        if not camera.isOpened():
            logger.error("无法打开摄像头")
            return False
            
        # 设置摄像头参数
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG['width'])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG['height'])
        camera.set(cv2.CAP_PROP_FPS, CAMERA_CONFIG['fps'])
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # 读取一帧测试
        success, frame = camera.read()
        if not success:
            logger.error("无法读取摄像头帧")
            return False
            
        logger.info(f"摄像头初始化成功: {frame.shape}")
        return True
        
    except Exception as e:
        logger.error(f"摄像头初始化失败: {str(e)}")
        return False

def process_pose_data(landmarks, image_shape):
    """处理姿态数据"""
    h, w = image_shape[:2]
    pose_data = []
    
    try:
        for idx in range(33):  # MediaPipe Pose 提供33个关键点
            landmark = landmarks[idx]
            if landmark.visibility > 0.5:
                pose_data.append([
                    float(landmark.x * w),  # 确保数据类型正确
                    float(landmark.y * h),
                    float(landmark.z * w),
                    float(landmark.visibility)
                ])
        
        if len(pose_data) >= 8:
            logger.debug(f"处理了 {len(pose_data)} 个关键点")
            return pose_data
        return None
    except Exception as e:
        logger.error(f"处理姿态数据失败: {e}")
        return None

def generate_frames(stream_type='original'):
    """优化的视频帧生成"""
    global camera, initial_frame, last_pose_update
    frame_count = 0
    last_frame_time = time.time()

    while True:
        try:
            if camera is None or not camera.isOpened():
                time.sleep(0.1)
                continue

            # 读取帧
            success, frame = camera.read()
            if not success:
                logger.error("读取摄像头帧失败")
                time.sleep(0.1)
                continue

            # 控制帧率
            current_time = time.time()
            if current_time - last_frame_time < 1.0 / CAMERA_CONFIG['fps']:
                continue
                
            last_frame_time = current_time
            frame_count += 1

            # 原始视频流
            if stream_type == 'original':
                try:
                    # 调整大小以提高性能
                    display_frame = cv2.resize(frame, (640, 480))
                    ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    else:
                        logger.error("编码视频帧失败")
                except Exception as e:
                    logger.error(f"处理视频帧失败: {e}")

            # 处理姿态数据
            if current_time - last_pose_update >= POSE_CONFIG['update_interval']:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = mp_manager.get_pose().process(frame_rgb)
                    
                    if results and results.pose_landmarks:
                        pose_data = process_pose_data(results.pose_landmarks.landmark, frame.shape)
                        if pose_data:
                            room_id = session.get('room')
                            if room_id and room_id in rooms:
                                logger.debug(f"发送姿态数据到房间: {room_id}")
                                socketio.emit('pose_data', {'data': pose_data}, room=room_id)
                                last_pose_update = current_time
                except Exception as e:
                    logger.error(f"处理姿态数据失败: {e}")

        except Exception as e:
            logger.error(f"生成帧失败: {e}")
            time.sleep(0.1)

# 基本路由
@app.route('/')
def index():
    """发送端页面"""
    return render_template('index.html')

@app.route('/receiver')
def receiver():
    """接收端页面"""
    room_id = request.args.get('room')
    if not room_id or room_id not in rooms:
        return "房间不存在", 404
    return render_template('receiver.html', room_id=room_id)

# Socket.IO 事件处理
@socketio.on('connect')
def handle_connect():
    logger.info(f'客户端已连接: {request.sid}')

@socketio.on('join_room')
def handle_join_room(data):
    try:
        room_id = data.get('room')
        role = data.get('role', 'receiver')
        
        logger.info(f"尝试加入房间: {room_id}, 角色: {role}, sid: {request.sid}")
        
        if not room_id:
            logger.error("未提供房间ID")
            emit('error', {'message': '未提供房间ID'})
            return
            
        if room_id not in rooms:
            if role == 'sender':
                logger.info(f"创建新房间: {room_id}")
                rooms[room_id] = Room(room_id)
            else:
                logger.error(f"房间不存在: {room_id}")
                emit('error', {'message': '房间不存在'})
                return
                
        room = rooms[room_id]
        
        # 如果是相同角色重新连接，先清除旧连接
        if role == 'sender' and room.sender:
            if room.sender == request.sid:
                logger.info(f"发送端重新连接: {room_id}")
            else:
                logger.error(f"房间已有发送端: {room_id}")
                emit('error', {'message': '房间已有发送端'})
                return
        elif role == 'receiver' and room.receiver:
            if room.receiver == request.sid:
                logger.info(f"接收端重新连接: {room_id}")
            else:
                logger.error(f"房间已有接收端: {room_id}")
                emit('error', {'message': '房间已有接收端'})
                return
        
        # 更新连接信息
        if role == 'sender':
            room.sender = request.sid
            logger.info(f"发送端加入房间: {room_id}")
        else:
            room.receiver = request.sid
            logger.info(f"接收端加入房间: {room_id}")
            
        join_room(room_id)
        session['room'] = room_id
        session['role'] = role
        
        # 如果是接收端且房间已经准备好，直接发送初始帧
        if role == 'receiver' and room.ready and room.initial_frame:
            logger.info(f"向新接收端发送已有初始帧")
            emit('initial_frame', {'image': room.initial_frame})
        
        emit('room_joined', {'room': room_id, 'role': role})
        logger.info(f"成功加入房间: {room_id}, 角色: {role}")
        
    except Exception as e:
        logger.error(f"加入房间失败: {str(e)}")
        emit('error', {'message': f'加入房间失败: {str(e)}'})

@socketio.on('initial_frame')
def handle_initial_frame(data):
    room_id = session.get('room')
    if not room_id or room_id not in rooms:
        return
        
    room = rooms[room_id]
    room.initial_frame = data['image']
    emit('initial_frame', {'image': data['image']}, room=room_id)

@socketio.on('pose_data')
def handle_pose_data(data):
    room_id = session.get('room')
    if not room_id or room_id not in rooms:
        return
        
    emit('pose_data', data, room=room_id)

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f'客户端断开连接: {request.sid}')
    room_id = session.get('room')
    if room_id and room_id in rooms:
        room = rooms[room_id]
        if request.sid == room.sender:
            logger.info(f'发送端离开房间: {room_id}')
            room.sender = None
        elif request.sid == room.receiver:
            logger.info(f'接收端离开房间: {room_id}')
            room.receiver = None
            
        if not room.sender and not room.receiver:
            logger.info(f'删除空房间: {room_id}')
            del rooms[room_id]
            
        leave_room(room_id)

# 添加视频相关路由
@app.route('/video_feed')
def video_feed():
    """视频流路由"""
    stream_type = request.args.get('stream', 'original')
    return Response(
        generate_frames(stream_type),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/start_capture', methods=['POST'])
def start_capture():
    """启动摄像头"""
    if initialize_camera():
        return jsonify({"status": "success", "message": "摄像头已启动"})
    return jsonify({"status": "error", "message": "摄像头启动失败"}), 500

@app.route('/stop_capture', methods=['POST'])
def stop_capture():
    """停止摄像头"""
    global camera
    try:
        if camera is not None:
            camera.release()
            camera = None
        return jsonify({"status": "success", "message": "摄像头已停止"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/capture_initial_frame', methods=['POST'])
def capture_initial_frame():
    """捕获初始帧"""
    global camera, initial_frame
    try:
        data = request.get_json()
        room_id = data.get('room_id')
        
        if not room_id:
            logger.error("请求中未包含房间ID")
            return jsonify({"status": "error", "message": "请求中未包含房间ID"}), 400
            
        if room_id not in rooms:
            logger.error(f"房间不存在: {room_id}")
            return jsonify({"status": "error", "message": "房间不存在"}), 404

        room = rooms[room_id]
        if not room.sender:
            logger.error(f"房间 {room_id} 没有发送端")
            return jsonify({"status": "error", "message": "房间没有发送端"}), 400

        if camera is None or not camera.isOpened():
            logger.error("摄像头未启动")
            return jsonify({"status": "error", "message": "摄像头未启动"}), 400
            
        success, frame = camera.read()
        if not success:
            logger.error("无法读取帧")
            return jsonify({"status": "error", "message": "无法读取帧"}), 500
            
        initial_frame = frame.copy()
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            img_data = base64.b64encode(buffer.tobytes()).decode()
            logger.info(f"发送初始帧到房间: {room_id}, 大小: {len(img_data)} bytes")
            
            # 保存到房间数据中并标记为就绪
            room.initial_frame = img_data
            room.ready = True
            logger.info("初始帧已保存到房间数据")
            
            # 发送给所有接收端
            socketio.emit('initial_frame', {'image': img_data}, room=room_id)
            
        return jsonify({"status": "success", "message": "初始帧已捕获"})
    except Exception as e:
        logger.error(f"捕获初始帧失败: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@socketio.on('get_initial_frame')
def handle_get_initial_frame(data):
    """处理获取初始帧请求"""
    room_id = data.get('room')
    if not room_id or room_id not in rooms:
        logger.error(f"获取初始帧失败: 房间 {room_id} 不存在")
        emit('error', {'message': '房间不存在'})
        return
        
    room = rooms[room_id]
    if room.ready and room.initial_frame:
        logger.info(f"发送初始帧到房间: {room_id}")
        emit('initial_frame', {'image': room.initial_frame})
    else:
        logger.info(f"房间 {room_id} 等待初始帧")
        emit('waiting_initial_frame')

if __name__ == "__main__":
    try:
        logger.info("正在启动服务器...")
        socketio.run(
            app,
            debug=False,
            host='0.0.0.0',
            port=5000,
            allow_unsafe_werkzeug=True
        )
    except Exception as e:
        logger.error(f"服务器启动失败: {str(e)}")
        sys.exit(1)
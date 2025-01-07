from flask import Flask, Response, jsonify, request, send_from_directory, render_template
import cv2
import numpy as np
import mediapipe as mp
import os
from werkzeug.utils import secure_filename
import time
import logging
from collections import deque
import json

# 获取项目根目录的绝对路径
project_root = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(project_root, 'templates')
static_dir = os.path.join(project_root, 'static')

app = Flask(__name__, 
           template_folder=template_dir,
           static_folder=static_dir)

# 配置上传文件的存储路径
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = os.path.join('static', 'models')
BACKGROUND_FOLDER = os.path.join('static', 'backgrounds')
ALLOWED_EXTENSIONS = {'gltf', 'glb', 'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# 确保必要的目录存在
for folder in [UPLOAD_FOLDER, MODEL_FOLDER, BACKGROUND_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# MediaPipe 初始化
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

# 定义面部点的颜色映射（使用浅色系）
FACE_COLORS = {
    'contour': [(255, 200, 200), (255, 220, 180), (255, 240, 160)],     # 浅蓝粉渐变
    'eyebrows': [(255, 180, 220), (255, 200, 200), (255, 220, 180)],    # 浅紫粉渐变
    'eyes': [(255, 220, 180), (255, 240, 160), (255, 255, 140)],        # 浅金蓝渐变
    'nose': [(255, 200, 180), (255, 220, 160), (255, 240, 140)],        # 浅橙蓝渐变
    'lips': [(255, 180, 200), (255, 200, 180), (255, 220, 160)],        # 浅红蓝渐变
    'general': [(255, 220, 200), (255, 240, 180), (255, 255, 160)]      # 浅青蓝渐变
}

# 定义手部连线样式
HAND_LANDMARKS_STYLE = mp_draw.DrawingSpec(
    color=(255, 220, 180),  # 浅蓝色点 (BGR)
    thickness=1,
    circle_radius=1
)
HAND_CONNECTIONS_STYLE = mp_draw.DrawingSpec(
    color=(255, 200, 160),  # 浅蓝线 (BGR)
    thickness=1
)

# 定义姿态连线样式
POSE_LANDMARKS_STYLE = mp_draw.DrawingSpec(
    color=(255, 240, 200),  # 浅天蓝点 (BGR)
    thickness=1,
    circle_radius=1
)
POSE_CONNECTIONS_STYLE = mp_draw.DrawingSpec(
    color=(255, 220, 180),  # 浅蓝线 (BGR)
    thickness=1
)

# 定义面部网格样式
FACE_MESH_STYLE = mp_draw.DrawingSpec(
    color=(255, 255, 255),  # 白色 (BGR)
    thickness=1,
    circle_radius=1
)
FACE_MESH_CONNECTIONS_STYLE = mp_draw.DrawingSpec(
    color=(255, 255, 255),  # 白色 (BGR)
    thickness=1
)

# 面部区域定义
FACE_REGIONS = {
    'contour': list(range(0, 17)),          # 脸部轮廓
    'right_eyebrow': list(range(17, 22)),   # 右眉毛
    'left_eyebrow': list(range(22, 27)),    # 左眉毛
    'nose': list(range(27, 36)),            # 鼻子
    'right_eye': list(range(36, 42)),       # 右眼
    'left_eye': list(range(42, 48)),        # 左眼
    'outer_lips': list(range(48, 60)),      # 外嘴唇
    'inner_lips': list(range(60, 68))       # 内嘴唇
}

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
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
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 全局变量
camera = None
current_frame = None
current_pose = None
current_hands = None
current_face = None

# 定义关键点连接
POSE_CONNECTIONS = [
    (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_EYE_INNER),
    (mp_pose.PoseLandmark.LEFT_EYE_INNER, mp_pose.PoseLandmark.LEFT_EYE),
    (mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.LEFT_EYE_OUTER),
    (mp_pose.PoseLandmark.RIGHT_EYE_INNER, mp_pose.PoseLandmark.RIGHT_EYE),
    (mp_pose.PoseLandmark.RIGHT_EYE, mp_pose.PoseLandmark.RIGHT_EYE_OUTER),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
    (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
]

# 定义面部点的样式（与手部保持一致）
FACE_LANDMARKS_STYLE = mp_draw.DrawingSpec(
    color=(255, 220, 180),  # 浅蓝色点 (BGR)
    thickness=1,
    circle_radius=1
)

# 定义面部网格连线样式（与手部保持一致）
FACE_CONNECTIONS_STYLE = mp_draw.DrawingSpec(
    color=(255, 200, 160),  # 浅蓝线 (BGR)
    thickness=1,
    circle_radius=1
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_camera():
    """摄像头初始化"""
    global camera
    try:
        if camera is not None:
            camera.release()
            
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            return False
            
        # 设置视频参数
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        camera.set(cv2.CAP_PROP_FPS, 30)
        
        # TODO: 添加更多视频参数配置
        # camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # 自动对焦
        # camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 自动曝光
        # camera.set(cv2.CAP_PROP_BRIGHTNESS, 1)  # 亮度
            
        return True
    except Exception as e:
        logging.error(f"摄像头初始化错误: {str(e)}")
        return False

@app.route('/start_capture', methods=['POST'])
def start_capture():
    """启动摄像头"""
    global camera
    if camera is not None and camera.isOpened():
        return jsonify({"status": "success"}), 200
        
    if init_camera():
        return jsonify({"status": "success"}), 200
    else:
        return jsonify({"status": "error"}), 500

@app.route('/stop_capture', methods=['POST'])
def stop_capture():
    """停止摄像头"""
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({"status": "success"}), 200

@app.route('/video_feed')
def video_feed():
    """视频流处理"""
    def generate():
        global camera, current_frame, current_pose, current_hands, current_face
        while True:
            if camera is None or not camera.isOpened():
                if not init_camera():
                    time.sleep(1)
                    continue
                    
            success, frame = camera.read()
            if not success:
                continue
                
            # 处理帧
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 姿态检测
            pose_results = pose.process(frame_rgb)
            if pose_results.pose_landmarks:
                current_pose = [[lm.x, lm.y, lm.z] for lm in pose_results.pose_landmarks.landmark]
                # 绘制姿态关键点和连线
                mp_draw.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    POSE_LANDMARKS_STYLE,
                    POSE_CONNECTIONS_STYLE
                )
            
            # 手部检测
            hands_results = hands.process(frame_rgb)
            if hands_results.multi_hand_landmarks:
                current_hands = []
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    current_hands.append([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    # 绘制手部关键点和连线
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        HAND_LANDMARKS_STYLE,
                        HAND_CONNECTIONS_STYLE
                    )
            
            # 面部网格检测
            face_results = face_mesh.process(frame_rgb)
            if face_results.multi_face_landmarks:
                current_face = [[lm.x, lm.y, lm.z] for lm in face_results.multi_face_landmarks[0].landmark]
                for face_landmarks in face_results.multi_face_landmarks:
                    # 绘制面部主要特征
                    mp_draw.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,  # 使用网格连接
                        landmark_drawing_spec=FACE_LANDMARKS_STYLE,  # 显示关键点
                        connection_drawing_spec=FACE_CONNECTIONS_STYLE  # 显示连接线
                    )
                    # 额外绘制眼睛和嘴巴轮廓
                    mp_draw.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,  # 眼睛轮廓
                        landmark_drawing_spec=None,  # 不显示额外的点
                        connection_drawing_spec=FACE_CONNECTIONS_STYLE
                    )

            current_frame = frame
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                   
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pose')
def get_pose():
    """获取当前姿态数据"""
    global current_pose, current_hands, current_face
    data = {
        "status": "success",
        "pose": current_pose if current_pose else None,
        "hands": current_hands if current_hands else None,
        "face": current_face if current_face else None
    }
    return jsonify(data)

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/room/<room_id>/models', methods=['GET'])
def get_room_models(room_id):
    """获取房间内的所有模型"""
    try:
        model_dir = os.path.join(MODEL_FOLDER, room_id)
        if not os.path.exists(model_dir):
            return jsonify({
                'status': 'success',
                'models': []
            })
            
        models = []
        for filename in os.listdir(model_dir):
            if filename.endswith(('.gltf', '.glb')):
                model_path = os.path.join('/static/models', room_id, filename)
                models.append({
                    'name': filename,
                    'url': model_path
                })
                
        return jsonify({
            'status': 'success',
            'models': models
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/room/<room_id>/models', methods=['POST'])
def upload_room_model(room_id):
    """上传模型到房间"""
    try:
        if 'model' not in request.files:
            return jsonify({
                'status': 'error',
                'message': '没有上传文件'
            }), 400
            
        file = request.files['model']
        if file.filename == '':
            return jsonify({
                'status': 'error', 
                'message': '未选择文件'
            }), 400
            
        if not file.filename.endswith(('.gltf', '.glb')):
            return jsonify({
                'status': 'error',
                'message': '不支持的文件格式'
            }), 400
            
        # 确保房间目录存在
        room_dir = os.path.join(MODEL_FOLDER, room_id)
        os.makedirs(room_dir, exist_ok=True)
        
        # 保存文件
        filename = secure_filename(file.filename)
        file_path = os.path.join(room_dir, filename)
        file.save(file_path)
        
        # 广播新模型通知
        # TODO: 实现WebSocket广播
        
        return jsonify({
            'status': 'success',
            'message': '模型上传成功',
            'model': {
                'name': filename,
                'url': os.path.join('/static/models', room_id, filename)
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/calibration', methods=['POST'])
def handle_calibration():
    """处理校准请求"""
    try:
        data = request.get_json()
        pose_type = data.get('pose_type')
        
        if not pose_type:
            return jsonify({
                'status': 'error',
                'message': '缺少姿势类型'
            }), 400
            
        # 获取当前帧的姿态数据
        pose_data = get_current_pose_data()  # 需要实现这个函数
        
        # 添加校准数据
        model_manager.add_calibration_pose(session['user_id'], pose_type, pose_data)
        
        return jsonify({
            'status': 'success',
            'message': f'已捕获 {pose_type} 姿势'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/recording/start', methods=['POST'])
def start_recording():
    """开始录制"""
    try:
        model_manager.start_recording(session['user_id'])
        return jsonify({
            'status': 'success',
            'message': '开始录制'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/recording/stop', methods=['POST'])
def stop_recording():
    """停止录制"""
    try:
        recording_data = model_manager.stop_recording(session['user_id'])
        return jsonify({
            'status': 'success',
            'message': '录制完成',
            'data': recording_data
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 
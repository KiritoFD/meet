from flask import Flask, Response, jsonify, request, send_from_directory, render_template
import cv2
import numpy as np
import mediapipe as mp
import os
from werkzeug.utils import secure_filename

# 获取项目根目录的绝对路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 全局变量
camera = None
current_frame = None
current_pose = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/start_capture', methods=['POST'])
def start_capture():
    global camera
    try:
        if camera is None:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                raise Exception("无法打开摄像头")
        return jsonify({"message": "摄像头已启动"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stop_capture', methods=['POST'])
def stop_capture():
    global camera
    try:
        if camera is not None:
            camera.release()
            camera = None
        return jsonify({"message": "摄像头已关闭"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def generate_frames():
    global current_frame, current_pose
    while True:
        if camera is None or not camera.isOpened():
            break
            
        success, frame = camera.read()
        if not success:
            break
            
        # 处理帧
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        # 存储当前姿态数据
        if results.pose_landmarks:
            current_pose = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
            
        # 绘制姿态标记点
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                
        current_frame = frame
        
        # 转换帧格式用于流式传输
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pose')
def get_pose():
    if current_pose is None:
        return jsonify([])
    return jsonify(current_pose)

@app.route('/upload_model', methods=['POST'])
def upload_model():
    if 'model' not in request.files:
        return jsonify({"error": "没有上传文件"}), 400
        
    file = request.files['model']
    if file.filename == '':
        return jsonify({"error": "没有选择文件"}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(MODEL_FOLDER, filename))
        return jsonify({"message": "模型上传成功", "filename": filename}), 200
    
    return jsonify({"error": "不支持的文件类型"}), 400

@app.route('/upload_background', methods=['POST'])
def upload_background():
    if 'background' not in request.files:
        return jsonify({"error": "没有上传文件"}), 400
        
    file = request.files['background']
    if file.filename == '':
        return jsonify({"error": "没有选择文件"}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(BACKGROUND_FOLDER, filename))
        return jsonify({"message": "背景上传成功", "filename": filename}), 200
    
    return jsonify({"error": "不支持的文件类型"}), 400

@app.route('/models/<path:filename>')
def serve_model(filename):
    return send_from_directory(MODEL_FOLDER, filename)

@app.route('/backgrounds/<path:filename>')
def serve_background(filename):
    return send_from_directory(BACKGROUND_FOLDER, filename)

@app.route('/')
def index():
    return render_template('display.html')

if __name__ == '__main__':
    app.run(debug=True)
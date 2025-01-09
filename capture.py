import cv2
import os
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import numpy as np
import io
import sys
import logging
import time

app = Flask(__name__)
CORS(app)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['BACKGROUND_FILE'] = 'uploads/background.jpg'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# OpenPose配置
try:
    from openpose import pyopenpose as op
    OPENPOSE_AVAILABLE = True
    # 配置OpenPose参数
    params = {
        "model_folder": "models/",
        "face": False,
        "hand": False,
        "net_resolution": "-1x368"
    }
    # 初始化OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    logger.info("OpenPose初始化成功")
except ImportError:
    OPENPOSE_AVAILABLE = False
    logger.warning("OpenPose未安装，将使用模拟数据")
    
# 全局变量
cap = None
latest_keypoints = None
background_image = None

def initialize_camera():
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("无法打开摄像头")
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return True
    return True

def simulate_pose_data():
    """生成模拟的姿势数据用于测试"""
    return [[[
        [320 + np.sin(time.time()) * 100, 240 + np.cos(time.time()) * 100, 0.9] 
        for _ in range(25)
    ]]]

def process_frame():
    global cap, latest_keypoints
    
    if not initialize_camera():
        return None
    
    ret, frame = cap.read()
    if not ret:
        logger.error("读取摄像头帧失败")
        return None
    
    if OPENPOSE_AVAILABLE:
        try:
            # 处理OpenPose
            datum = op.Datum()
            datum.cvInputData = frame
            opWrapper.emplaceAndPop([datum])
            
            # 更新关键点数据
            if datum.poseKeypoints is not None:
                latest_keypoints = datum.poseKeypoints.tolist()
                return datum.cvOutputData
        except Exception as e:
            logger.error(f"OpenPose处理错误: {str(e)}")
            return frame
    else:
        # 如果OpenPose不可用，返回原始帧并使用模拟数据
        latest_keypoints = simulate_pose_data()
        return frame

@app.route('/pose', methods=['GET'])
def get_pose():
    global latest_keypoints
    if latest_keypoints is None and not OPENPOSE_AVAILABLE:
        latest_keypoints = simulate_pose_data()
    return jsonify(latest_keypoints if latest_keypoints is not None else [])

@app.route('/start_capture', methods=['POST'])
def start_capture():
    if initialize_camera():
        return jsonify({"message": "Camera initialized", "openpose_available": OPENPOSE_AVAILABLE})
    return jsonify({"error": "Failed to initialize camera"}), 500

@app.route('/upload_background', methods=['POST'])
def upload_background():
    global background_image
    if 'background' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['background']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'background.jpg')
        file.save(filepath)
        background_image = cv2.imread(filepath)
        return jsonify({"message": "Background uploaded successfully"}), 200
    except Exception as e:
        logger.error(f"背景上传错误: {str(e)}")
        return jsonify({"error": "Failed to save background"}), 500

@app.route('/get_frame')
def get_frame():
    frame = process_frame()
    if frame is None:
        return jsonify({"error": "Failed to capture frame"}), 500
    
    try:
        # 将帧转换为JPEG格式
        _, buffer = cv2.imencode('.jpg', frame)
        return send_file(
            io.BytesIO(buffer.tobytes()),
            mimetype='image/jpeg'
        )
    except Exception as e:
        logger.error(f"帧处理错误: {str(e)}")
        return jsonify({"error": "Failed to process frame"}), 500

def cleanup():
    global cap
    if cap is not None:
        cap.release()
        cap = None

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        cleanup()

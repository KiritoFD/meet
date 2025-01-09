import os
import sys
from flask import Flask, Response, render_template, jsonify, send_from_directory
import cv2
import mediapipe as mp
import numpy as np
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 获取项目根目录的绝对路径
project_root = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(project_root, 'templates')
static_dir = os.path.join(project_root, 'static')

app = Flask(__name__, 
           template_folder=template_dir,
           static_folder=static_dir)

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
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# 全局变量
camera = None
current_frame = None
current_pose = None

@app.route('/')
def index():
    return render_template('display.html')

@app.route('/start_capture', methods=['POST'])
def start_capture():
    global camera
    try:
        if camera is not None:
            camera.release()
            
        camera = cv2.VideoCapture(0)
        
        # 调整摄像头参数以获得更好的画质
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 增加分辨率
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # 自动曝光
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not camera.isOpened():
            raise Exception("无法打开摄像头")
            
        # 预热摄像头
        for _ in range(5):
            ret, _ = camera.read()
            if not ret:
                raise Exception("摄像头预热失败")
                
        logger.info("摄像头已成功启动")
        return jsonify({"message": "摄像头已启动", "status": "success"}), 200
    except Exception as e:
        logger.error(f"启动摄像头失败: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route('/stop_capture', methods=['POST'])
def stop_capture():
    global camera
    try:
        if camera is not None:
            camera.release()
            camera = None
            logger.info("摄像头已关闭")
        return jsonify({"message": "摄像头已关闭", "status": "success"}), 200
    except Exception as e:
        logger.error(f"关闭摄像头失败: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 500

def generate_frames():
    global camera, current_frame, current_pose
    
    # 定义上半身的连接关系
    POSE_CONNECTIONS = [
        # 面部关键点
        (0, 1), (1, 2), (2, 3), (3, 4),    # 左侧面部
        (0, 4), (4, 5), (5, 6), (6, 7),    # 右侧面部
        (8, 9), (9, 10),                    # 嘴部
        (0, 5),                             # 眉心连接
        (1, 2), (2, 3),                     # 左眉
        (4, 5), (5, 6),                     # 右眉
        (2, 5),                             # 鼻梁
        (3, 6),                             # 眼睛连接
        
        # 身体关键点
        (11, 12),                           # 肩膀连接
        (11, 13), (13, 15),                 # 左臂
        (12, 14), (14, 16),                 # 右臂
        
        # 左手指连接
        (15, 17), (17, 19), (19, 21),       # 左手拇指
        (15, 17), (17, 19), (19, 21),       # 左手食指
        (15, 17), (17, 19), (19, 21),       # 左手中指
        (15, 17), (17, 19), (19, 21),       # 左手无名指
        (15, 17), (17, 19), (19, 21),       # 左手小指
        
        # 右手指连接
        (16, 18), (18, 20), (20, 22),       # 右手拇指
        (16, 18), (18, 20), (20, 22),       # 右手食指
        (16, 18), (18, 20), (20, 22),       # 右手中指
        (16, 18), (18, 20), (20, 22),       # 右手无名指
        (16, 18), (18, 20), (20, 22),       # 右手小指
        
        # 手指横向连接
        (17, 19), (19, 21),                 # 左手指节连接
        (18, 20), (20, 22),                 # 右手指节连接
        
        # 躯干
        (11, 23), (12, 24), (23, 24)        # 上身躯干
    ]
    
    # 更新关键点列表
    upper_body_points = [
        # 面部关键点
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        # 身体关键点
        11, 12, 13, 14, 15, 16,
        # 手指关键点
        17, 18, 19, 20, 21, 22,             # 基础手指点
        23, 24                              # 躯干点
    ]

    # 定义面部关键连接，更详细的版本
    FACE_CONNECTIONS = [
        # 眉毛
        ([70, 63, 105, 66, 107, 55, 65], (0, 0, 255)),          # 左眉
        ([336, 296, 334, 293, 300, 285, 295], (0, 0, 255)),     # 右眉
        
        # 眼睛
        ([33, 246, 161, 160, 159, 158, 157, 173, 133], (255, 0, 0)),  # 左眼
        ([362, 398, 384, 385, 386, 387, 388, 466, 263], (255, 0, 0)), # 右眼
        
        # 鼻子
        ([168, 6, 197, 195, 5], (0, 255, 0)),        # 鼻梁
        ([198, 209, 49, 48, 219], (0, 255, 0)),      # 鼻翼左
        ([420, 432, 279, 278, 438], (0, 255, 0)),    # 鼻翼右
        
        # 嘴唇
        ([61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291], (0, 255, 255)),  # 上唇
        ([146, 91, 181, 84, 17, 314, 405, 321, 375, 291], (0, 255, 255)),    # 下唇
        
        # 面部轮廓关键点
        ([10, 338, 297, 332, 284], (255, 255, 0)),   # 左脸
        ([454, 323, 361, 288, 397], (255, 255, 0)),  # 右脸
        ([152, 148, 176], (255, 255, 0)),            # 下巴
    ]

    while True:
        try:
            if camera is None or not camera.isOpened():
                logger.warning("摄像头未打开或已断开")
                continue

            success, frame = camera.read()
            if not success or frame is None:
                continue

            # 调整图像大小以提高性能
            frame = cv2.resize(frame, (1280, 720))
            
            # 镜像翻转，使显示更自然
            frame = cv2.flip(frame, 1)
            
            # 提高图像对比度
            frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 处理姿势
            pose_results = pose.process(frame_rgb)
            
            # 创建一个半透明的遮罩层用于绘制
            overlay = frame.copy()
            
            # 绘制姿势关键点
            if pose_results.pose_landmarks:
                h, w, c = frame.shape
                
                # 绘制连接线，使用更粗的线条和更鲜艳的颜色
                for connection in POSE_CONNECTIONS:
                    try:
                        start_point = pose_results.pose_landmarks.landmark[connection[0]]
                        end_point = pose_results.pose_landmarks.landmark[connection[1]]
                        
                        if start_point.visibility > 0.7 and end_point.visibility > 0.7:
                            start_x = int(start_point.x * w)
                            start_y = int(start_point.y * h)
                            end_x = int(end_point.x * w)
                            end_y = int(end_point.y * h)
                            
                            # 使用更鲜艳的颜色
                            cv2.line(overlay, 
                                   (start_x, start_y), 
                                   (end_x, end_y),
                                   (0, 255, 255), 
                                   3)  # 加粗线条
                            
                            # 在关键点处添加光晕效果
                            cv2.circle(overlay, (start_x, start_y), 5, (255, 255, 0), -1)
                            cv2.circle(overlay, (end_x, end_y), 5, (255, 255, 0), -1)
                    except Exception as e:
                        continue
            
            # 将遮罩层与原始帧混合
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            
            # 添加帧率显示
            fps = camera.get(cv2.CAP_PROP_FPS)
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
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
    global camera
    is_running = camera is not None and camera.isOpened()
    return jsonify({
        "isRunning": is_running,
        "status": "running" if is_running else "stopped"
    })

if __name__ == "__main__":
    # 确保必要的目录存在
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print(f"服务器启动在 http://localhost:5000")
    print(f"模板目录: {template_dir}")
    print(f"静态文件目录: {static_dir}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
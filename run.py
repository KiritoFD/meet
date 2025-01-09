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
            camera.release()  # 确保先释放之前的摄像头
        
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise Exception("无法打开摄像头")
            
        # 设置摄像头参数
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        
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
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                continue

            success, frame = camera.read()
            if not success or frame is None:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 处理姿势
            pose_results = pose.process(frame_rgb)
            
            # 处理手部
            hands_results = hands.process(frame_rgb)

            # 处理面部
            face_results = face_mesh.process(frame_rgb)

            # 绘制姿势关键点
            if pose_results.pose_landmarks:
                h, w, c = frame.shape
                
                # 绘制连接线
                for connection in POSE_CONNECTIONS:
                    try:
                        start_point = pose_results.pose_landmarks.landmark[connection[0]]
                        end_point = pose_results.pose_landmarks.landmark[connection[1]]
                        
                        if start_point.visibility > 0.5 and end_point.visibility > 0.5:
                            start_x = int(start_point.x * w)
                            start_y = int(start_point.y * h)
                            end_x = int(end_point.x * w)
                            end_y = int(end_point.y * h)
                            
                            # 根据连接类型使用不同颜色
                            if connection[0] <= 10:  # 面部连接
                                color = (255, 0, 0)  # 蓝色
                                thickness = 1
                            elif connection[0] >= 15 and connection[0] <= 22:  # 手指连接
                                color = (0, 255, 255)  # 黄色
                                thickness = 1
                                # 添加手指关节点
                                cv2.circle(frame, (start_x, start_y), 2, color, -1)
                                cv2.circle(frame, (end_x, end_y), 2, color, -1)
                            else:  # 其他连接
                                color = (0, 255, 0)  # 绿色
                                thickness = 2
                            
                            cv2.line(frame, (start_x, start_y), (end_x, end_y), 
                                   color=color, thickness=thickness)
                    except Exception as e:
                        continue

                # 绘制关键点
                for idx in upper_body_points:
                    try:
                        landmark = pose_results.pose_landmarks.landmark[idx]
                        if landmark.visibility > 0.5:
                            cx = int(landmark.x * w)
                            cy = int(landmark.y * h)
                            
                            if idx <= 10:  # 面部关键点
                                color = (255, 0, 0)  # 蓝色
                                radius = 2
                            elif idx in [11, 12, 13, 14, 15, 16]:  # 手臂关键点
                                color = (0, 0, 255)  # 红色
                                radius = 3
                            elif idx >= 17:  # 手部关键点
                                color = (0, 255, 255)  # 黄色
                                radius = 2
                            else:  # 躯干关键点
                                color = (255, 255, 0)  # 青色
                                radius = 3
                            
                            cv2.circle(frame, (cx, cy), radius, color, -1)
                            cv2.circle(frame, (cx, cy), radius + 1, color, 1)
                    except Exception as e:
                        continue

            # 绘制手部关键点和连接
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    # 绘制手部关键点和连接线
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # 自定义手指连接线
                    h, w, c = frame.shape
                    
                    # 定义手指关键点组
                    fingers = [
                        [4, 3, 2, 1],    # 拇指
                        [8, 7, 6, 5],    # 食指
                        [12, 11, 10, 9],  # 中指
                        [16, 15, 14, 13], # 无名指
                        [20, 19, 18, 17]  # 小指
                    ]
                    
                    # 绘制每个手指的连接线
                    for finger in fingers:
                        for i in range(len(finger)-1):
                            start = hand_landmarks.landmark[finger[i]]
                            end = hand_landmarks.landmark[finger[i+1]]
                            
                            start_x = int(start.x * w)
                            start_y = int(start.y * h)
                            end_x = int(end.x * w)
                            end_y = int(end.y * h)
                            
                            # 绘制连接线
                            cv2.line(frame, 
                                   (start_x, start_y), 
                                   (end_x, end_y),
                                   (0, 255, 255),  # 黄色
                                   2)
                            
                            # 绘制关节点
                            cv2.circle(frame, (start_x, start_y), 3, (0, 255, 255), -1)
                            cv2.circle(frame, (end_x, end_y), 3, (0, 255, 255), -1)
                    
                    # 绘制手指横向连接
                    knuckles = [5, 9, 13, 17]  # 指关节点
                    for i in range(len(knuckles)-1):
                        start = hand_landmarks.landmark[knuckles[i]]
                        end = hand_landmarks.landmark[knuckles[i+1]]
                        
                        start_x = int(start.x * w)
                        start_y = int(start.y * h)
                        end_x = int(end.x * w)
                        end_y = int(end.y * h)
                        
                        cv2.line(frame, 
                               (start_x, start_y), 
                               (end_x, end_y),
                               (0, 255, 255),  # 黄色
                               1)

            # 绘制面部网格
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    h, w, c = frame.shape
                    
                    # 绘制所有面部关键点，使用更柔和的颜色
                    for i in range(468):  # MediaPipe Face Mesh 有468个关键点
                        landmark = face_landmarks.landmark[i]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        
                        # 使用更柔和的颜色方案
                        if i in range(0, 68):  # 轮廓点
                            color = (200, 180, 130)  # 淡金色
                        elif i in range(68, 136):  # 眉毛点
                            color = (180, 120, 90)  # 深棕色
                        elif i in range(136, 204):  # 眼睛点
                            color = (120, 150, 230)  # 淡蓝色
                        elif i in range(204, 272):  # 鼻子点
                            color = (150, 200, 180)  # 青绿色
                        else:  # 嘴唇和其他点
                            color = (140, 160, 210)  # 淡紫色
                        
                        # 绘制更小的点，提高精致感
                        cv2.circle(frame, (x, y), 1, color, -1)
                    
                    # 主要特征连接线使用更优雅的颜色
                    feature_colors = {
                        'eyebrow': (160, 140, 110),   # 眉毛：深金色
                        'eye': (130, 160, 220),       # 眼睛：天蓝色
                        'nose': (140, 190, 170),      # 鼻子：青色
                        'mouth': (170, 150, 200),     # 嘴唇：淡紫色
                        'face': (190, 170, 120)       # 轮廓：金棕色
                    }
                    
                    # 绘制主要连接线
                    for points, _ in FACE_CONNECTIONS:
                        points_coords = []
                        for point_idx in points:
                            landmark = face_landmarks.landmark[point_idx]
                            x = int(landmark.x * w)
                            y = int(landmark.y * h)
                            points_coords.append((x, y))
                            
                            # 根据点的位置选择颜色
                            if point_idx in range(68, 136):  # 眉毛区域
                                color = feature_colors['eyebrow']
                            elif point_idx in range(136, 204):  # 眼睛区域
                                color = feature_colors['eye']
                            elif point_idx in range(204, 272):  # 鼻子区域
                                color = feature_colors['nose']
                            elif point_idx > 272:  # 嘴唇区域
                                color = feature_colors['mouth']
                            else:  # 面部轮廓
                                color = feature_colors['face']
                            
                            # 绘制稍大的关键点
                            cv2.circle(frame, (x, y), 2, color, -1)
                        
                        # 绘制连接线
                        for i in range(len(points_coords)-1):
                            cv2.line(frame, points_coords[i], points_coords[i+1], color, 1)
                    
                    # 移除文字标注，保持界面简洁

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
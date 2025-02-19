import cv2
import mediapipe as mp
import numpy as np
import logging
import os
import time
from pose.multi_detector import MultiDetector
from pose.pose_binding import PoseBinding
from pose.pose_deformer import PoseDeformer
from camera.manager import CameraManager
from config.settings import CAMERA_CONFIG
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO
from pose.types import PoseData, Landmark  # 添加这行导入

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 获取项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(project_root, 'frontend', 'pages')  # 修改这一行
static_dir = os.path.join(project_root, 'frontend', 'static')

# 初始化组件
camera_manager = CameraManager(config=CAMERA_CONFIG)
detector = MultiDetector()
pose_binding = PoseBinding()
pose_deformer = PoseDeformer()

# MediaPipe 初始化
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh  # 添加 face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles  # 添加绘制样式

# 初始化检测器
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=True,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

face_mesh = mp_face_mesh.FaceMesh(  # 添加 face_mesh 初始化
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 初始化 Flask
app = Flask(__name__, 
           template_folder=template_dir,  # 使用修改后的路径
           static_folder=static_dir,
           static_url_path='/static')
socketio = SocketIO(app, cors_allowed_origins="*")

# 全局变量
deformed_frame = None  # 添加这行来存储最新的变形结果
reference_frame = None
reference_pose = None
regions = None

class FrameProcessor:
    """帧处理器类，用于集中管理帧处理状态"""
    def __init__(self):
        self.reference_frame = None
        self.reference_pose = None
        self.regions = None
        self.deformed_frame = None
        self.height = None
        self.width = None

# 创建全局帧处理器实例
frame_processor = FrameProcessor()

def create_display_window():
    """创建垂直堆叠的显示窗口和控制按钮"""
    cv2.namedWindow('Control Panel', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Deformed', cv2.WINDOW_NORMAL)
    
    # 调整窗口位置和大小
    cv2.resizeWindow('Control Panel', 400, 100)
    cv2.moveWindow('Control Panel', 50, 0)
    cv2.moveWindow('Original', 50, 150)
    cv2.moveWindow('Deformed', 50, 500)
    
    # 创建控制面板图像
    control_panel = np.zeros((100, 400, 3), dtype=np.uint8)
    cv2.putText(control_panel, "C: Capture  R: Reset  Q: Quit", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow('Control Panel', control_panel)

def handle_mouse_events(event, x, y, flags, param):
    """处理鼠标事件的回调函数"""
    if event == cv2.EVENT_LBUTTONDOWN:
        if y < 50:  # 假设按钮区域在上半部分
            capture_reference_frame(param)

def resize_frame(frame, max_height=300):
    """保持宽高比调整图像大小"""
    height, width = frame.shape[:2]
    if height > max_height:
        ratio = max_height / height
        new_width = int(width * ratio)
        frame = cv2.resize(frame, (new_width, max_height))
    return frame

def process_landmarks(pose_results, face_results=None):
    """处理姿态和面部检测结果，创建 PoseData 对象"""
    if not pose_results or not pose_results.pose_landmarks:
        return None
        
    # 处理姿态关键点
    landmarks = []
    for lm in pose_results.pose_landmarks.landmark:
        landmarks.append(Landmark(
            x=lm.x,
            y=lm.y,
            z=lm.z,
            visibility=getattr(lm, 'visibility', 1.0)
        ))
        
    # 处理面部关键点
    face_landmarks = []
    if face_results and face_results.multi_face_landmarks:
        for face_landmark in face_results.multi_face_landmarks[0].landmark:
            face_landmarks.append(Landmark(
                x=face_landmark.x,
                y=face_landmark.y,
                z=face_landmark.z,
                visibility=1.0  # 面部关键点默认可见度为1.0
            ))
            
    return PoseData(
        landmarks=landmarks,
        face_landmarks=face_landmarks,
        timestamp=time.time(),
        confidence=sum(lm.visibility for lm in landmarks) / len(landmarks)
    )

def generate_original_frames():
    """生成原始视频帧"""
    while True:
        if not camera_manager.is_running:
            time.sleep(0.1)
            continue
            
        frame = camera_manager.read_frame()
        if frame is None:
            continue
            
        # 处理姿态检测和绘制
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(frame_rgb)
        display_frame = frame.copy()
        
        if pose_results.pose_landmarks:
            # 使用 MediaPipe 的连接定义绘制更完整的姿态
            mp_drawing.draw_landmarks(
                display_frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0),
                    thickness=2,
                    circle_radius=2
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(255, 0, 0),
                    thickness=2
                )
            )
        
        # 转换帧格式
        ret, buffer = cv2.imencode('.jpg', display_frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate_deformed_frames():
    """生成变形后的视频帧"""
    while True:
        if not camera_manager.is_running:
            time.sleep(0.1)
            continue
            
        if frame_processor.deformed_frame is None:
            # 如果没有变形结果，生成空白帧
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, buffer = cv2.imencode('.jpg', blank)
        else:
            ret, buffer = cv2.imencode('.jpg', frame_processor.deformed_frame)
            
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """渲染主页"""
    return render_template('display.html')

@app.route('/video_feed')
def video_feed():
    """原始视频流"""
    return Response(
        generate_original_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/deformed_feed')
def deformed_feed():
    """变形视频流"""
    return Response(
        generate_deformed_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/start_capture', methods=['POST'])
def start_capture():
    """启动摄像头"""
    try:
        success = camera_manager.start()
        return jsonify({'success': success})
    except Exception as e:
        logger.error(f"启动摄像头失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/stop_capture', methods=['POST'])
def stop_capture():
    """停止摄像头"""
    try:
        success = camera_manager.stop()
        return jsonify({'success': success})
    except Exception as e:
        logger.error(f"停止摄像头失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/check_stream_status')
def check_stream_status():
    """获取流状态"""
    try:
        status = {
            'video': {
                'is_streaming': camera_manager.is_running,
                'fps': camera_manager.current_fps,
                'frame_count': camera_manager.frame_count
            },
            'audio': {
                'is_recording': False,  # 暂时不处理音频
                'sample_rate': 0,
                'buffer_size': 0
            }
        }
        return jsonify(status)
    except Exception as e:
        logger.error(f"获取流状态失败: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/capture_reference', methods=['POST'])
def capture_reference():
    """捕获参考帧"""
    try:
        # 首先检查摄像头状态
        if not camera_manager.is_running:
            return jsonify({
                'success': False,
                'message': '摄像头未运行'
            }), 400
            
        # 获取摄像头画面
        frame = camera_manager.read_frame()
        
        # 检查是否获取到帧
        if frame is None:  # 完全无法获取画面
            return jsonify({
                'success': False,
                'message': '无法获取摄像头画面'
            }), 500
            
        # 检查帧是否有效
        if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:  # 空帧
            return jsonify({
                'success': False,
                'message': '无效的摄像头画面'
            }), 500
            
        # 检测姿态
        pose_results = pose.process(frame)
        if not pose_results or not pose_results.pose_landmarks:
            return jsonify({
                'success': False,
                'message': '未检测到人物姿态'
            }), 400
            
        # 检测面部
        face_results = face_mesh.process(frame)
        if not face_results or not face_results.multi_face_landmarks:
            return jsonify({
                'success': False,
                'message': '未检测到面部'
            }), 400  # 确保返回元组
            
        # 创建姿态数据
        pose_data = process_landmarks(pose_results, face_results)
        if not pose_data:
            return jsonify({
                'success': False,
                'message': '处理姿态数据失败'
            }), 500  # 确保返回元组
            
        # 检查姿态数据的完整性
        if len(pose_data.landmarks) < 33:
            return jsonify({
                'success': False,
                'message': '检测到的关键点不完整'
            }), 400  # 确保返回元组
            
        # 检查关键点可见度
        visible_points = [lm for lm in pose_data.landmarks if lm.visibility > 0.5]
        if len(visible_points) < 15:
            return jsonify({
                'success': False,
                'message': '姿态检测置信度过低'
            }), 400  # 确保返回元组
            
        # 创建绑定区域
        regions = pose_binding.create_binding(frame, pose_data)
        if not regions:
            return jsonify({
                'success': False,
                'message': '创建绑定区域失败'
            }), 500  # 确保返回元组
            
        # 保存参考帧和姿态数据
        frame_processor.reference_frame = frame.copy()
        frame_processor.reference_pose = pose_data
        frame_processor.regions = regions
            
        # 返回成功结果
        return jsonify({
            'success': True,
            'details': {
                'regions_info': {
                    'body': len([r for r in regions if r.type == 'body']),
                    'face': len([r for r in regions if r.type == 'face'])
                },
                'reference_frame': frame.tolist()
            }
        }), 200  # 确保返回元组
            
    except Exception as e:
        logger.error(f"捕获失败: {e}")
        return jsonify({
            'success': False,
            'message': f'捕获失败: {str(e)}'
        }), 500  # 确保返回元组

@app.route('/test')
def test_page():
    """渲染测试页面"""
    return render_template('test_capture.html')

@app.route('/reset_capture', methods=['POST'])
def reset_capture():
    """重置捕获状态"""
    try:
        frame_processor.reference_frame = None
        frame_processor.reference_pose = None
        frame_processor.regions = None
        frame_processor.deformed_frame = None
        
        return jsonify({
            'success': True,
            'message': '重置成功'
        })
    except Exception as e:
        logger.error(f"重置失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

def process_frame(frame):
    """处理单帧图像"""
    global frame_processor
    
    if frame is None:
        return None
        
    # 使用多模型检测器处理帧
    detection_result = detector.process_frame(frame)
    if detection_result is None:
        return None
    
    # 绘制检测结果
    display_frame = detector.draw_detections(frame, detection_result)
    
    # 如果有参考帧，执行变形
    if frame_processor.reference_frame is not None:
        try:
            # 更新绑定并变形
            updated_regions = pose_binding.update_binding(
                frame_processor.regions, 
                detection_result
            )
            
            if updated_regions:
                deformed = pose_deformer.deform(
                    frame_processor.reference_frame,
                    frame_processor.reference_pose,
                    frame,
                    detection_result,
                    updated_regions
                )
                
                if deformed is not None:
                    # 在变形结果上显示检测结果
                    frame_processor.deformed_frame = detector.draw_detections(
                        deformed, 
                        detection_result
                    )
                    
        except Exception as e:
            logger.error(f"变形处理失败: {str(e)}")
    
    return display_frame

def main():
    # 全局变量
    reference_frame = None
    reference_pose = None
    regions = None
    
    # 创建显示窗口和按钮
    create_display_window()
    
    # 设置鼠标回调
    param_dict = {
        'reference_frame': None,
        'reference_pose': None,
        'regions': None,
        'pose_binding': pose_binding,
        'current_frame': None,
        'current_pose': None
    }
    cv2.setMouseCallback('Control Panel', handle_mouse_events, param_dict)
    
    # 启动摄像头
    if not camera_manager.start():
        logger.error("无法启动摄像头")
        return
    
    logger.info("系统已启动")
    logger.info("点击按钮或按键:")
    logger.info("- C: 捕获参考帧")
    logger.info("- R: 重置参考帧")
    logger.info("- Q: 退出程序")
    
    try:
        while True:
            frame = camera_manager.read_frame()
            if frame is None:
                continue
                
            # 调整显示大小
            frame = resize_frame(frame)
            display_frame = frame.copy()
            
            # 处理姿态
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(frame_rgb)
            
            if pose_results.pose_landmarks:
                # 显示姿态关键点
                mp_drawing.draw_landmarks(
                    display_frame,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )
                
                # 转换关键点格式
                pose_data = process_landmarks(pose_results)
                
                # 如果有参考帧，执行变形
                if reference_frame is not None and pose_data is not None:
                    try:
                        # 更新绑定并变形
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
                                deformed_frame = resize_frame(deformed_frame)
                                # 在变形结果上显示姿态
                                mp_drawing.draw_landmarks(
                                    deformed_frame,
                                    pose_results.pose_landmarks,
                                    mp_pose.POSE_CONNECTIONS
                                )
                                cv2.imshow('Deformed', deformed_frame)
                    except Exception as e:
                        logger.error(f"变形处理失败: {str(e)}")
            
            # 更新当前帧信息到参数字典
            if pose_results.pose_landmarks:
                param_dict['current_frame'] = frame.copy()
                param_dict['current_pose'] = pose_data
            
            # 显示原始画面
            cv2.imshow('Original', display_frame)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("用户退出程序")
                break
            elif key == ord('c'):
                if param_dict['current_pose'] is not None:
                    capture_reference_frame(param_dict)
            elif key == ord('r'):
                reset_reference_frame(param_dict)
                logger.info("已重置参考帧")
                
    except Exception as e:
        logger.error(f"程序运行错误: {str(e)}")
    finally:
        camera_manager.stop()
        cv2.destroyAllWindows()
        pose.close()
        logger.info("程序已结束")

def capture_reference_frame(param_dict):
    """捕获参考帧的通用函数"""
    if param_dict['current_pose'] is not None:
        param_dict['reference_frame'] = param_dict['current_frame'].copy()
        param_dict['reference_pose'] = param_dict['current_pose']
        param_dict['regions'] = param_dict['pose_binding'].create_binding(
            param_dict['reference_frame'], 
            param_dict['reference_pose']
        )
        logger.info(f"已捕获参考帧，创建了 {len(param_dict['regions'])} 个绑定区域")
    else:
        logger.warning("未检测到姿态，无法捕获参考帧")

def reset_reference_frame(param_dict):
    """重置参考帧的通用函数"""
    param_dict['reference_frame'] = None
    param_dict['reference_pose'] = None
    param_dict['regions'] = None

if __name__ == "__main__":
    try:
        # 添加全局变量初始化
        reference_frame = None
        reference_pose = None
        regions = None
        
        socketio.run(app, host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序出错: {e}")
        logger.exception("程序异常退出")
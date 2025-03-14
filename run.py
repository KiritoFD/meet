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
import absl.logging
import asyncio

# Fix import paths
from camera.manager import CameraManager
from pose.drawer import PoseDrawer  
from config import settings
from config.settings import CAMERA_CONFIG, POSE_CONFIG
from pose.pose_binding import PoseBinding
from pose.detector import PoseDetector
from pose.types import PoseData
from pose.sender import PoseSender  # Add this line to import PoseSender

# Make sure the nvidia module is in the Python path
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Then import from nvidia module
from nvidia.model_manager import NVIDIAModelManager
from nvidia.network_simulator import NetworkSimulator
from nvidia.keypoint_compressor import KeypointCompressor
from nvidia.keypoint_receiver import KeypointReceiver
from nvidia.keypoint_stream import KeypointStreamHandler

# 抑制 TensorFlow 警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
logging.getLogger('tensorflow').setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)

# 禁用 mediapipe 的调试日志
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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


# 定义上传文件夹路径
UPLOAD_FOLDER = os.path.join(project_root, 'uploads')

# 初始化 Socket.IO
socketio = SocketIO(app, cors_allowed_origins="*")

# 在全局变量部分添加NVIDIA模型
nvidia_model_manager = NVIDIAModelManager.get_instance()
network_simulator = NetworkSimulator(profile="medium")  # 默认使用中等网络环境
keypoint_compressor = KeypointCompressor(precision=2)

# 初始化关键点流处理器
keypoint_receiver = KeypointReceiver()
keypoint_stream_handler = KeypointStreamHandler(keypoint_receiver)

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
camera_manager = CameraManager(config=CAMERA_CONFIG)
pose_drawer = PoseDrawer()
pose_binding = PoseBinding()
initial_frame = None
initial_regions = None


# 初始化检测器
pose_detector = PoseDetector()

# 在全局变量部分添加
REFERENCE_DIR = os.path.join(project_root, 'output', 'reference')
os.makedirs(REFERENCE_DIR, exist_ok=True)

from pose.initial_manager import InitialFrameManager
initial_manager = InitialFrameManager(os.path.join(project_root, 'output'))

from stream.stream_manager import StreamManager
from stream.http_stream import HTTPStreamHandler
from config.stream_config import DEFAULT_STREAM_CONFIG, HIGH_QUALITY_STREAM_CONFIG, LOW_BANDWIDTH_STREAM_CONFIG

# 在全局变量部分更新
stream_manager = StreamManager(CAMERA_CONFIG)
http_streamer = HTTPStreamHandler(stream_manager)

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

@app.route('/')
def index():
    """渲染显示页面"""
    return render_template('display.html')

@app.route('/start_capture', methods=['POST'])
def start_capture():
    """启动摄像头"""
    try:
        if camera_manager.is_running:
            return jsonify({'success': False, 'message': 'Camera is already running'})
        success = camera_manager.start()
        if success:
            return jsonify({'success': True, 'resolution': {'width': camera_manager.width, 'height': camera_manager.height}})
        return jsonify({'success': False, 'error': 'Failed to start camera'}), 500
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

@app.route('/video_feed')
def video_feed():
    """视频流路由"""
    return Response(http_streamer.generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream/high_quality')
def high_quality_stream():
    """高质量视频流"""
    high_quality_streamer = HTTPStreamHandler(stream_manager, HIGH_QUALITY_STREAM_CONFIG)
    return Response(high_quality_streamer.generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream/low_bandwidth')
def low_bandwidth_stream():
    """低带宽视频流"""
    low_bandwidth_streamer = HTTPStreamHandler(stream_manager, LOW_BANDWIDTH_STREAM_CONFIG)
    return Response(low_bandwidth_streamer.generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream_info')
def stream_info():
    """获取流信息"""
    return jsonify(http_streamer.get_stream_info())



@app.route('/check_stream_status')
def check_stream_status():
    try:
        # 获取原始数据
        status = {
            'video': {
                'is_streaming': camera_manager.is_running,
                'fps': camera_manager.current_fps,
                'frame_count': stream_manager.frame_count,
                'resolution': {
                    'width': camera_manager.width,
                    'height': camera_manager.height
                } if camera_manager.is_running else None,
                'frame_rate': camera_manager.frame_rate
            },
            'audio': {
                #'is_recording': audio_processor.is_recording,
                #'sample_rate': audio_processor.sample_rate,
                #'buffer_size': len(audio_processor.frames) if hasattr(audio_processor, 'frames') else 0
            }
        }
        
        # 添加NVIDIA模型状态
        if stream_manager:
            status['nvidia_model'] = {
                'enabled': stream_manager.use_nvidia_model,
                'initialized': nvidia_model_manager.is_initialized
            }
        
        # 添加网络模拟器状态
        if network_simulator:
            status['network'] = network_simulator.get_status()
            
            # 添加带宽使用估算
            if hasattr(stream_manager, 'last_pose_data') and stream_manager.last_pose_data:
                bandwidth_estimate = keypoint_compressor.estimate_bandwidth(
                    stream_manager.last_pose_data, 
                    fps=camera_manager.current_fps or 30
                )
                status['network']['bandwidth_estimate'] = bandwidth_estimate
        
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"获取流状态失败: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/capture_initial', methods=['POST'])
def capture_initial():
    """捕获初始参考帧"""
    try:
        # 1. 检查相机状态
        if not camera_manager.is_running:
            return jsonify({
                'success': False, 
                'error': 'Camera is not running'
            }), 400
            
        # 2. 捕获图像
        success, frame = camera_manager.read()
        if not success or frame is None:
            return jsonify({
                'success': False,
                'error': 'Failed to capture frame'
            }), 500
        
        # 3. 检测姿态
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(frame_rgb)
        
        if not pose_results or not pose_results.pose_landmarks:
            return jsonify({
                'success': False,
                'error': 'No pose detected'
            }), 400
            
        # 4. 准备姿态数据
        try:
            keypoints = PoseDetector.mediapipe_to_keypoints(pose_results.pose_landmarks)
            pose_data = PoseData(
                keypoints=keypoints,
                timestamp=time.time(),
                confidence=1.0
            )
            
            # 5. 保存参考帧并创建绑定
            result = initial_manager.save_initial_frame(frame, pose_data)
            
            # 6. 设置流管理器的参考帧
            stream_manager.set_reference(frame, pose_data)
            
            # 7. 尝试设置NVIDIA模型的参考帧
            if nvidia_model_manager.is_initialized:
                nvidia_model_manager.set_reference_frame(frame)
                stream_manager.use_nvidia_model = True
                logger.info("NVIDIA模型参考帧已设置")
                
            # 8. 为关键点接收器设置参考帧
            keypoint_receiver.set_reference_frame(frame)
            logger.info("关键点接收器参考帧已设置")
            
            return jsonify({
                'success': True,
                'path': result,
                'frame_size': {
                    'width': frame.shape[1],
                    'height': frame.shape[0]
                }
            })
            
        except Exception as e:
            logger.error(f"处理关键点失败: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
        
    except Exception as e:
        logger.error(f"捕获初始帧失败: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/reference_status', methods=['GET'])
def get_reference_status():
    """获取参考帧状态"""
    try:
        status = initial_manager.get_status()
        return jsonify({
            'success': True,
            **status
        })
    except Exception as e:
        logger.error(f"获取参考帧状态失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/keypoint_video_feed')
def keypoint_video_feed():
    """基于关键点的视频流"""
    return Response(keypoint_stream_handler.generate_stream(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/keypoint_stream/start', methods=['POST'])
def start_keypoint_stream():
    """启动关键点流"""
    try:
        # 确保有参考帧
        if not hasattr(keypoint_receiver, 'reference_frame') or keypoint_receiver.reference_frame is None:
            if stream_manager and stream_manager.reference_frame is not None:
                keypoint_receiver.set_reference_frame(stream_manager.reference_frame)
            else:
                return jsonify({
                    'success': False,
                    'error': '未设置参考帧'
                }), 400
        
        # 启动关键点流
        success = keypoint_stream_handler.start()
        
        # 启用演示模式（可选）
        demo_mode = request.json.get('demo_mode', False) if request.json else False
        keypoint_stream_handler.enable_demo_mode(demo_mode)
        
        return jsonify({
            'success': success,
            'status': keypoint_stream_handler.get_status()
        })
    except Exception as e:
        logger.error(f"启动关键点流失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/keypoint_stream/stop', methods=['POST'])
def stop_keypoint_stream():
    """停止关键点流"""
    try:
        keypoint_stream_handler.stop()
        return jsonify({
            'success': True
        })
    except Exception as e:
        logger.error(f"停止关键点流失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/keypoint_stream/status')
def keypoint_stream_status():
    """获取关键点流状态"""
    try:
        return jsonify({
            'success': True,
            'status': keypoint_stream_handler.get_status()
        })
    except Exception as e:
        logger.error(f"获取关键点流状态失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/keypoint_stream/send', methods=['POST'])
def send_keypoint_data():
    """发送关键点数据"""
    try:
        data = request.json
        
        if not data or not data.get('keypoints'):
            return jsonify({
                'success': False,
                'error': '缺少关键点数据'
            }), 400
        
        # 处理关键点数据
        success = keypoint_stream_handler.process_keypoint_data(data)
        
        return jsonify({
            'success': success
        })
    except Exception as e:
        logger.error(f"发送关键点数据失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/keypoint_stream/demo', methods=['POST'])
def toggle_demo_mode():
    """切换演示模式"""
    try:
        data = request.json
        enable = data.get('enable', True)
        
        keypoint_stream_handler.enable_demo_mode(enable)
        
        return jsonify({
            'success': True,
            'demo_mode': keypoint_stream_handler.demo_mode
        })
    except Exception as e:
        logger.error(f"切换演示模式失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def generate_frames():
    """生成视频帧"""
    while True:
        if not camera_manager.is_running:
            time.sleep(0.1)
            continue
            
        frame = camera_manager.read_frame()
        if frame is None:
            continue
            
        # 转换颜色空间
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # 处理姿态
            pose_results = pose.process(frame_rgb)
            # 处理手部
            hands_results = hands.process(frame_rgb)
            # 处理面部
            face_results = face_mesh.process(frame_rgb)
            
            # 合并所有关键点数据
            landmarks_data = {
                'pose': [],
                'face': [],
                'left_hand': [],
                'right_hand': []
            }
            
            # 添加姿态关键点
            if pose_results.pose_landmarks:
                for landmark in pose_results.pose_landmarks.landmark:
                    landmarks_data['pose'].append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
            
            # 添加面部关键点
            if face_results.multi_face_landmarks:
                for landmark in face_results.multi_face_landmarks[0].landmark:
                    landmarks_data['face'].append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })
            
            # 添加手部关键点
            if hands_results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                    # 确定是左手还是右手
                    handedness = hands_results.multi_handedness[hand_idx].classification[0].label
                    hand_type = 'left_hand' if handedness == 'Left' else 'right_hand'
                    
                    for landmark in hand_landmarks.landmark:
                        landmarks_data[hand_type].append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z
                        })
            
            # 发送所有关键点数据
            if any(landmarks_data.values()):
                socketio.emit('pose_data', landmarks_data)
                logger.info(f"发送关键点数据: 姿态={len(landmarks_data['pose'])}, "
                          f"面部={len(landmarks_data['face'])}, "
                          f"左手={len(landmarks_data['left_hand'])}, "
                          f"右手={len(landmarks_data['right_hand'])} 个关键点")
            
        except Exception as e:
            logger.error(f"处理关键点时出错: {str(e)}")
            continue
            
        # 转换帧格式用于传输
        try:
            ret, buffer = cv2.imencode('.jpg', frame)  # 直接使用原始帧
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            logger.error(f"编码帧时出错: {str(e)}")

@app.route('/camera_status')
def camera_status():
    """获取摄像头状态"""
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
    """处理客户端连接"""
    logger.info("客户端已连接")
    # Ensure pose_sender is defined before using it
    global pose_sender
    pose_sender = PoseSender()
    pose_sender.connect(socketio)

@socketio.on('disconnect')
def handle_disconnect():
    """处理客户端断开连接"""
    logger.info("客户端已断开")
    pose_sender.disconnect()



@app.errorhandler(Exception)
def handle_error(error):
    """全局错误处理"""
    logger.error(f"发生错误: {str(error)}")
    return jsonify({
        'success': False,
        'error': str(error)
    }), 500

@app.route('/camera/settings', methods=['GET', 'POST'])
def camera_settings():
    """获取或更新相机设置"""
    if request.method == 'GET':
        return jsonify(camera_manager.get_settings())
        
    settings = request.json
    success = camera_manager.update_settings(settings)
    return jsonify({'success': success})

@app.route('/camera/reset', methods=['POST'])
def reset_camera():
    """重置相机设置"""
    success = camera_manager.reset_settings()
    return jsonify({'success': success})

@app.route('/status')
def get_status():
    """获取当前状态"""
    try:
        status = {
            'camera': {
                'isActive': camera_manager.is_running,
                'fps': camera_manager.current_fps
            },
            'simulator': {
                'isActive': network_simulator.is_running if network_simulator else False,
                'profile': network_simulator.profile if network_simulator else None,
                'stats': network_simulator.get_status() if network_simulator and network_simulator.is_running else {}
            }
        }
        return jsonify(status)
    except Exception as e:
        logger.error(f"获取状态失败: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/nvidia/status')
def nvidia_status():
    """获取NVIDIA模型状态"""
    try:
        status = nvidia_model_manager.get_status()
        return jsonify({
            'success': True,
            'status': status
        })
    except Exception as e:
        logger.error(f"获取NVIDIA模型状态失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/nvidia/toggle', methods=['POST'])
def toggle_nvidia_model():
    """切换是否使用NVIDIA模型"""
    try:
        data = request.json
        enable = data.get('enable', True)
        
        # 确保流管理器已初始化
        if not stream_manager:
            return jsonify({
                'success': False,
                'error': '流管理器未初始化'
            }), 500
        
        success = stream_manager.toggle_nvidia_model(enable)
        return jsonify({
            'success': success,
            'enabled': stream_manager.use_nvidia_model if success else False
        })
    except Exception as e:
        logger.error(f"切换NVIDIA模型失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/nvidia/initialize', methods=['POST'])
def initialize_nvidia_model():
    """初始化NVIDIA模型"""
    try:
        # 获取可选的模型路径参数
        data = request.json or {}
        checkpoint_path = data.get('checkpoint_path')
        
        # 初始化模型
        success = nvidia_model_manager.initialize(checkpoint_path=checkpoint_path)
        
        # 如果成功初始化，更新流管理器设置
        if success and stream_manager:
            stream_manager.toggle_nvidia_model(True)
        
        return jsonify({
            'success': success,
            'status': nvidia_model_manager.get_status()
        })
    except Exception as e:
        logger.error(f"初始化NVIDIA模型失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/send_keypoints', methods=['POST'])
def send_keypoints():
    """接收关键点数据并通过网络模拟器模拟传输"""
    try:
        data = request.json
        
        if not data or not data.get('keypoints'):
            return jsonify({
                'success': False,
                'error': '缺少关键点数据'
            }), 400
            
        # 压缩关键点数据
        compressed_data = keypoint_compressor.compress_pose_data(
            PoseData(
                keypoints=data.get('keypoints', []),
                timestamp=data.get('timestamp', time.time()),
                confidence=data.get('confidence', 1.0)
            )
        )
        
        # 序列化数据
        serialized = keypoint_compressor.serialize_for_transmission(compressed_data)
        data_size = len(serialized)
        
        # 模拟网络传输
        if not network_simulator.is_running:
            network_simulator.start()
            
        transmission_success = network_simulator.simulate_send(data_size)
        
        # 如果传输成功，执行NVIDIA模型动画生成
        if transmission_success and nvidia_model_manager.is_initialized and stream_manager.reference_frame is not None:
            # 反序列化数据
            decompressed_data = keypoint_compressor.decompress_pose_data(compressed_data)
            
            if decompressed_data:
                # 使用NVIDIA模型生成动画
                animated_frame = nvidia_model_manager.animate(stream_manager.reference_frame, decompressed_data)
                
                # 保存最近的姿态数据用于带宽估算
                stream_manager.last_pose_data = decompressed_data
                stream_manager.last_compressed_size = data_size
                
                # 更新网络统计
                stream_manager.network_stats['transmitted_frames'] += 1
                stream_manager.network_stats['total_bytes'] += data_size
                
                return jsonify({
                    'success': True,
                    'transmitted': True,
                    'data_size': data_size,
                    'network_status': network_simulator.get_status()
                })
        else:
            # 传输失败或模型未初始化
            stream_manager.network_stats['dropped_frames'] += 1
            
            return jsonify({
                'success': True,
                'transmitted': False,
                'data_size': data_size,
                'error': '传输失败或模型未初始化' if not transmission_success else '模型未初始化',
                'network_status': network_simulator.get_status()
            })
            
    except Exception as e:
        logger.error(f"处理关键点数据失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"处理失败: {str(e)}"
        }), 500

@app.route('/network/set_profile', methods=['POST'])
def set_network_profile():
    """设置网络配置文件"""
    try:
        data = request.json
        profile = data.get('profile', 'medium')
        
        if not network_simulator:
            return jsonify({
                'success': False,
                'error': '网络模拟器未初始化'
            }), 500
            
        success = network_simulator.set_profile(profile)
        
        return jsonify({
            'success': success,
            'profile': profile if success else None,
            'status': network_simulator.get_status() if success else None
        })
        
    except Exception as e:
        logger.error(f"设置网络配置文件失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/network/status')
def network_status():
    """获取网络模拟器状态"""
    try:
        if not network_simulator:
            return jsonify({
                'success': False,
                'error': '网络模拟器未初始化'
            }), 500
            
        status = network_simulator.get_status()
        
        # 添加估计的带宽使用情况
        if stream_manager and stream_manager.last_pose_data:
            bandwidth_estimate = keypoint_compressor.estimate_bandwidth(
                stream_manager.last_pose_data,
                fps=30  # 假设30fps
            )
            status['bandwidth_estimate'] = bandwidth_estimate
            
        # 添加模拟器接收和播放统计信息
        if stream_manager:
            status['playback_stats'] = stream_manager.network_stats
            
        return jsonify({
            'success': True,
            'status': status
        })
        
    except Exception as e:
        logger.error(f"获取网络状态失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@socketio.on('pose_data')
def handle_pose_data(data):
    """处理从前端发送的姿态数据"""
    try:
        # 创建PoseData对象
        pose_data = PoseData(
            keypoints=data.get('pose', []),
            timestamp=time.time(),
            confidence=1.0
        )
        
        # 压缩数据
        compressed_data = keypoint_compressor.compress_pose_data(pose_data)
        serialized = keypoint_compressor.serialize_for_transmission(compressed_data)
        data_size = len(serialized)
        
        # 模拟网络传输
        if not network_simulator.is_running:
            network_simulator.start()
            
        transmission_success = network_simulator.simulate_send(data_size)
        
        # 如果传输成功且NVIDIA模型已初始化，则使用模型生成动画
        if transmission_success and nvidia_model_manager.is_initialized and stream_manager.reference_frame is not None:
            # 记录统计信息
            stream_manager.last_pose_data = pose_data
            stream_manager.last_compressed_size = data_size
            stream_manager.network_stats['transmitted_frames'] += 1
            stream_manager.network_stats['total_bytes'] += data_size
            
            # 生成动画帧
            animated_frame = nvidia_model_manager.animate(stream_manager.reference_frame, pose_data)
            
            # 如果需要，可以在这里将生成的帧发送回前端
        else:
            # 传输失败统计
            stream_manager.network_stats['dropped_frames'] += 1
            
    except Exception as e:
        logger.error(f"处理Socket姿态数据失败: {str(e)}")

# 添加一个带宽监控回调
def on_bandwidth_update(stats):
    """带宽更新回调"""
    try:
        socketio.emit('bandwidth_update', {
            'bandwidth_kbps': stats['bandwidth_kbps'],
            'usage_kbps': stats['usage_kbps'],
            'packet_loss': stats['packet_loss'],
            'latency_ms': stats['latency_ms']
        })
    except Exception as e:
        logger.error(f"带宽更新回调失败: {str(e)}")

# 在全局变量初始化部分修改
def init_network_simulator():
    """初始化网络模拟器"""
    global network_simulator
    network_simulator = NetworkSimulator(profile="medium")
    network_simulator.register_callback(on_bandwidth_update)
    network_simulator.start()
    logger.info("网络模拟器已初始化")

def init_pose_system():
    """初始化姿态处理系统"""
    try:
        # 初始化姿态检测器
        logger.info("正在初始化姿态检测器...")
        pose_detector = PoseDetector()
        
        # 初始化姿态绑定器
        logger.info("正在初始化姿态绑定器...")
        pose_binding = PoseBinding()
        
        # 初始化绘制器
        logger.info("正在初始化姿态绘制器...")
        pose_drawer = PoseDrawer()
        
        # 尝试初始化NVIDIA模型
        try:
            logger.info("正在初始化NVIDIA模型...")
            nvidia_model_manager.initialize()
            logger.info("NVIDIA模型初始化完成")
        except Exception as e:
            logger.warning(f"NVIDIA模型初始化失败（可忽略）: {str(e)}")
        
        # 初始化网络模拟器
        try:
            init_network_simulator()
        except Exception as e:
            logger.warning(f"初始化网络模拟器失败: {str(e)}")
        
        return pose_detector, pose_binding, pose_drawer
        
    except Exception as e:
        logger.error(f"姿态系统初始化失败: {str(e)}")
        raise

async def setup_jitsi():
    # transport = JitsiTransport(JITSI_CONFIG)
    # meeting_manager = JitsiMeetingManager(JITSI_CONFIG)
    
    return None, None

async def main():
    # ... 其他代码 ...
    
    # 注释掉 Jitsi 相关的初始化和设置
    '''
    # 初始化 Jitsi 会议管理器
    meeting_manager = JitsiMeetingManager(JITSI_CONFIG)
    await meeting_manager.start()
    
    try:
        default_room_id = "default_room"
        host_id = "host_1"
        room_id = await meeting_manager.create_meeting(
            room_id=default_room_id,
            host_id=host_id
        )
        logger.info(f"Created default meeting room: {room_id}")
    except Exception as e:
        logger.error(f"Failed to create default meeting room: {e}")
        raise
    '''
    
    try:
        # 直接使用 Flask 的 run 方法
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True  # 开发模式
        )
    except Exception as e:
        logger.error(f"Failed to start web server: {e}")
        raise
    finally:
        pass
        # await meeting_manager.stop()  # 注释掉

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 抑制 TensorFlow 和 Mediapipe 警告
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    absl.logging.set_verbosity(absl.logging.ERROR)
    logging.getLogger('mediapipe').setLevel(logging.ERROR)
    
    try:
        # 创建必要的目录
        os.makedirs(os.path.join(project_root, 'models'), exist_ok=True)  # 为NVIDIA模型创建目录
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        
        # 运行主程序
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序出错: {e}")
        logger.exception("程序异常退出")
    finally:
        # 清理资源
        cv2.destroyAllWindows()
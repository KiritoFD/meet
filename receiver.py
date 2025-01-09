import cv2
import numpy as np
import socketio
import logging
import json
import zlib
import base64
from flask import Flask, render_template, Response
from flask_socketio import SocketIO

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# 全局变量
initial_image = None
current_pose = None
output_frame = None

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

class PoseTransformManager:
    def __init__(self, max_cache_size=30):
        self.last_matrix = None
        self.last_result = None
        self.pose_cache = []
        self.max_cache_size = max_cache_size
        
    def update_cache(self, pose_data):
        """更新姿态缓存"""
        self.pose_cache.append(pose_data)
        if len(self.pose_cache) > self.max_cache_size:
            self.pose_cache.pop(0)
            
    def get_smoothed_pose(self):
        """获取平滑后的姿态数据"""
        if not self.pose_cache:
            return None
            
        # 使用最近的几帧计算平均姿态
        recent_poses = self.pose_cache[-5:]
        if not recent_poses:
            return self.pose_cache[-1]
            
        avg_pose = []
        for i in range(len(recent_poses[0])):
            point_data = np.mean([pose[i] for pose in recent_poses], axis=0)
            avg_pose.append(point_data.tolist())
            
        return avg_pose

# 创建全局管理器实例
pose_manager = PoseTransformManager()

def decompress_pose_data(compressed_data):
    """解压缩姿态数据"""
    try:
        binary_data = base64.b64decode(compressed_data)
        decompressed = zlib.decompress(binary_data)
        return json.loads(decompressed.decode())
    except Exception as e:
        logger.error(f"解压缩姿态数据失败: {e}")
        return None

def apply_pose_transform(image, pose_data):
    """改进的姿态变换算法"""
    if image is None or not pose_data or len(pose_data) < 8:
        return None
        
    try:
        h, w = image.shape[:2]
        
        # 1. 提取关键点和权重
        points = np.float32([[p[0], p[1]] for p in pose_data])
        depths = np.float32([p[2] for p in pose_data])
        weights = np.float32([p[3] for p in pose_data])
        
        # 2. 计算面部方向
        face_center = np.mean(points[:11], axis=0, weights=weights[:11])
        face_normal = np.cross(
            points[7] - points[8],    # 耳朵连线
            points[11] - points[12]   # 肩膀连线
        )
        face_normal = face_normal / np.linalg.norm(face_normal)
        
        # 3. 构建透视变换矩阵
        src_points = np.float32([
            points[0],     # 鼻子
            points[7],     # 左耳
            points[8],     # 右耳
            points[11],    # 左肩
            points[12]     # 右肩
        ])
        
        # 4. 计算目标点（考虑深度）
        scale_factor = 1.0 + depths.mean() * 0.1
        dst_points = src_points.copy()
        dst_points = dst_points * scale_factor
        
        # 5. 计算透视变换矩阵
        M = cv2.findHomography(
            src_points, 
            dst_points,
            cv2.RANSAC,
            5.0,
            confidence=0.99,
            maxIters=2000
        )[0]
        
        # 6. 应用平滑
        if hasattr(apply_pose_transform, 'last_matrix'):
            alpha = 0.8  # 增加平滑系数
            M = cv2.addWeighted(
                apply_pose_transform.last_matrix,
                1-alpha,
                M,
                alpha,
                0
            )
        apply_pose_transform.last_matrix = M.copy()
        
        # 7. 应用变换
        result = cv2.warpPerspective(
            image,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        # 8. 添加稳定性检查
        if hasattr(apply_pose_transform, 'last_result'):
            diff = cv2.absdiff(result, apply_pose_transform.last_result)
            change_ratio = np.mean(diff) / 255.0
            
            if change_ratio > 0.1:  # 如果变化太大
                result = cv2.addWeighted(
                    apply_pose_transform.last_result,
                    0.7,
                    result,
                    0.3,
                    0
                )
        
        apply_pose_transform.last_result = result.copy()
        return result
        
    except Exception as e:
        logger.error(f"姿态变换失败: {e}")
        if hasattr(apply_pose_transform, 'last_result'):
            return apply_pose_transform.last_result
        return image

def generate_frames():
    """生成视频帧"""
    global initial_image, current_pose, output_frame
    
    while True:
        if initial_image is not None and current_pose is not None:
            try:
                # 应用姿态变换
                frame = apply_pose_transform(initial_image, current_pose)
                if frame is not None:
                    output_frame = frame
                    
                    # 编码并发送帧
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                logger.error(f"生成帧失败: {e}")
        
        # 如果没有数据，显示黑屏
        if output_frame is None:
            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, buffer = cv2.imencode('.jpg', black_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """主页"""
    return render_template('receiver.html')

@app.route('/video_feed')
def video_feed():
    """视频流路由"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def handle_connect():
    """处理WebSocket连接"""
    logger.info('客户端已连接')

@socketio.on('disconnect')
def handle_disconnect():
    """处理WebSocket断开"""
    logger.info('客户端已断开')

@socketio.on('initial_frame')
def handle_initial_frame(data):
    """处理初始帧数据"""
    global initial_image
    try:
        # 解码图像数据
        nparr = np.frombuffer(base64.b64decode(data['image']), np.uint8)
        initial_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        logger.info('已接收初始帧')
    except Exception as e:
        logger.error(f"处理初始帧失败: {e}")

@socketio.on('pose_data')
def handle_pose_data(data):
    """处理姿态数据"""
    global current_pose, pose_manager
    try:
        pose_data = decompress_pose_data(data['data'])
        if pose_data:
            pose_manager.update_cache(pose_data)
            current_pose = pose_manager.get_smoothed_pose()
            logger.debug('已接收并处理姿态数据')
    except Exception as e:
        logger.error(f"处理姿态数据失败: {e}")

if __name__ == '__main__':
    logger.info("接收端启动中...")
    socketio.run(app, debug=False, host='0.0.0.0', port=5001) 
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import logging
from pose.drawer import PoseDrawer
from .manager import ReceiveManager
from .transform import PoseTransformer
from .static import static_bp
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReceiveApp:
    def __init__(self):
        template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend', 'pages')
        static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend', 'static')
        
        self.app = Flask(__name__,
                        template_folder=template_dir,
                        static_folder=static_dir,
                        static_url_path='/static')
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.manager = ReceiveManager()
        self.transformer = PoseTransformer()
        
        # 注册蓝图
        self.app.register_blueprint(static_bp)
        
        self._setup_routes()
        self._setup_socket_handlers()
    
    def _setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('receiver.html')
            
        @self.app.route('/video_feed')
        def video_feed():
            return Response(
                self.manager.generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
    
    def _setup_socket_handlers(self):
        @self.socketio.on('connect')
        def handle_connect():
            logger.info('客户端已连接')
            
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info('客户端已断开')
            
        @self.socketio.on('pose_data')
        def handle_pose_data(data):
            self.manager.handle_pose_data(data)
            
        @self.socketio.on('initial_frame')
        def handle_initial_frame(data):
            self.manager.handle_initial_frame(data)
    
    def run(self, host='0.0.0.0', port=5001):
        logger.info(f"接收端启动在 http://{host}:{port}")
        self.socketio.run(self.app, host=host, port=port, debug=False)

if __name__ == '__main__':
    app = ReceiveApp()
    app.run() 
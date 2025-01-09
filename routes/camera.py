from flask import jsonify
from utils.logger import logger
from core.camera import CameraManager

# 创建摄像头管理器实例
camera_manager = CameraManager()

def init_camera_routes(app):
    @app.route('/camera_status', methods=['GET'])
    def get_camera_status():
        """获取摄像头状态"""
        is_active = camera_manager.is_active()
        return jsonify({
            "isActive": is_active
        })

    @app.route('/start_capture', methods=['POST'])
    def start_capture():
        """启动摄像头"""
        try:
            if camera_manager.initialize():
                logger.info("摄像头启动成功")
                return jsonify({"status": "success", "message": "摄像头已启动"})
            else:
                logger.error("摄像头启动失败")
                return jsonify({"status": "error", "message": "摄像头启动失败"}), 500
        except Exception as e:
            logger.error(f"启动摄像头出错: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route('/stop_capture', methods=['POST'])
    def stop_capture():
        """停止摄像头"""
        try:
            camera_manager.release()
            return jsonify({"status": "success", "message": "摄像头已停止"})
        except Exception as e:
            logger.error(f"停止摄像头出错: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 500 
# ...existing imports...
from pose.verification_manager import VerificationManager

class MeetApp:
    def __init__(self):
        # ...existing init code...
        self.verification_manager = VerificationManager()

    @app.route('/verify_identity', methods=['POST'])
    def verify_identity(self):
        try:
            # 获取当前帧
            if not self.camera or not self.camera.is_running:
                return jsonify({
                    'success': False,
                    'message': '请先启动摄像头'
                })

            frame = self.camera.read_frame()
            if frame is None:
                return jsonify({
                    'success': False,
                    'message': '无法获取摄像头画面'
                })

            # 验证身份
            result = self.verification_manager.verify_identity(frame)
            
            return jsonify({
                'success': True,
                'verification': {
                    'passed': result.success,
                    'message': result.message,
                    'confidence': result.confidence
                }
            })
            
        except Exception as e:
            logger.error(f"身份验证失败: {str(e)}")
            return jsonify({
                'success': False,
                'message': str(e)
            })

    @app.route('/capture_reference', methods=['POST'])
    def capture_reference(self):
        try:
            if not self.camera or not self.camera.is_running:
                return jsonify({
                    'success': False,
                    'message': '请先启动摄像头'
                })

            frame = self.camera.read_frame()
            if frame is None:
                return jsonify({
                    'success': False,
                    'message': '无法获取摄像头画面'
                })

            # 捕获参考帧
            result = self.verification_manager.capture_reference(frame)
            
            return jsonify({
                'success': True,
                'verification': {
                    'passed': result.success,
                    'message': result.message,
                    'confidence': result.confidence
                }
            })
            
        except Exception as e:
            logger.error(f"捕获参考帧失败: {str(e)}")
            return jsonify({
                'success': False,
                'message': str(e)
            })

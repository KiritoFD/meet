import sys
import os
from pathlib import Path

# 调试信息
project_root = str(Path(__file__).parent.parent)
print("Python path in test:", sys.path)
print("Current file:", __file__)
print("Project root:", project_root)
print("Directory contents:", os.listdir(project_root))

# 添加项目根目录到Python路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
import cv2
import numpy as np
import time
import logging
import mediapipe as mp
from flask import Flask
from flask_socketio import SocketIO

# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    from connect.socket_manager import SocketManager
    from connect.pose_sender import PoseSender
    from connect.pose_protocol import PoseProtocol
    from pose.drawer import PoseDrawer
except ImportError as e:
    logger.error(f"Import Error: {e}")
    logger.error(f"Looking for modules in: {project_root}")
    logger.error(f"Directory contents: {os.listdir(project_root)}")
    raise

class TestSystem:
    @pytest.fixture
    def setup_system(self):
        """初始化测试环境"""
        try:
            # 初始化 Flask 和 SocketIO
            app = Flask(__name__)
            socketio = SocketIO(app, cors_allowed_origins="*")
            logger.info("Flask and SocketIO initialized")
            
            socket_manager = SocketManager(socketio)
            logger.info("SocketManager initialized")
            
            # 初始化 MediaPipe
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("MediaPipe Pose initialized")
            
            # 初始化组件
            pose_drawer = PoseDrawer()
            pose_sender = PoseSender(socketio, socket_manager)
            logger.info("Components initialized")
            
            # 连接服务器
            if not socket_manager.connect():
                raise ConnectionError("Failed to connect to server")
            logger.info("Connected to server")
            
            components = {
                'pose': pose,
                'drawer': pose_drawer,
                'sender': pose_sender,
                'socket': socket_manager,
                'socketio': socketio,
                'app': app
            }
            
            yield components
            
            # 清理
            logger.info("Cleaning up...")
            pose.close()
            socket_manager.disconnect()
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise

    def test_system_initialization(self, setup_system):
        """测试系统初始化"""
        try:
            assert setup_system['pose'] is not None, "MediaPipe Pose 初始化失败"
            assert setup_system['drawer'] is not None, "PoseDrawer 初始化失败"
            assert setup_system['sender'] is not None, "PoseSender 初始化失败"
            assert setup_system['socket'] is not None, "SocketManager 初始化失败"
            logger.info("System initialization test passed")
        except AssertionError as e:
            logger.error(f"Initialization test failed: {e}")
            raise

    def test_pose_detection(self, setup_system):
        """测试姿态检测"""
        try:
            pose = setup_system['pose']
            
            # 创建测试帧
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.circle(test_frame, (320, 240), 50, (255, 255, 255), -1)  # 添加一个白色圆形
            
            # 转换颜色空间并处理
            frame_rgb = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            # 详细验证检测结果
            assert results is not None, "姿态检测返回None"
            logger.info("Pose detection test passed")
            
        except Exception as e:
            logger.error(f"Pose detection test failed: {e}")
            raise

    def test_pose_drawing(self, setup_system):
        """测试姿态绘制"""
        try:
            pose = setup_system['pose']
            drawer = setup_system['drawer']
            
            # 创建测试帧
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.circle(test_frame, (320, 240), 50, (255, 255, 255), -1)
            
            # 处理帧
            frame_rgb = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            # 绘制姿态
            processed_frame = drawer.draw_frame(
                test_frame.copy(),
                results,
                None,  # hands_results
                None   # face_results
            )
            
            # 详细验证输出
            assert processed_frame is not None, "绘制结果为None"
            assert processed_frame.shape == test_frame.shape, "输出帧尺寸不匹配"
            assert np.any(processed_frame != test_frame), "绘制没有改变原始帧"
            logger.info("Pose drawing test passed")
            
        except Exception as e:
            logger.error(f"Pose drawing test failed: {e}")
            raise

    def test_pose_sending(self, setup_system):
        """测试姿态数据发送"""
        try:
            pose = setup_system['pose']
            sender = setup_system['sender']
            socket = setup_system['socket']
            
            # Add timeout for connection check
            connection_timeout = time.time() + 5  # 5 seconds timeout
            while not socket.connected and time.time() < connection_timeout:
                time.sleep(0.1)
            
            if not socket.connected:
                raise ConnectionError("Socket connection timeout")
            
            # 创建测试帧
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.circle(test_frame, (320, 240), 50, (255, 255, 255), -1)
            
            # 处理帧
            frame_rgb = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            # 测试发送
            success = sender.send_pose_data(
                room="test_room",
                pose_results=results,
                timestamp=time.time()
            )
            assert success, "姿态数据发送失败"
            logger.info("Pose sending test passed")
            
        except Exception as e:
            logger.error(f"Pose sending test failed: {e}")
            raise

    def test_performance_monitoring(self, setup_system):
        """测试性能监控"""
        sender = setup_system['sender']
        
        # 开始监控
        sender.start_monitoring()
        
        # 执行一些操作
        for _ in range(100):
            sender.send_pose_data(
                room="test_room",
                pose_results=self._generate_test_pose()
            )
        
        # 获取性能报告
        stats = sender.get_stats()
        
        # 验证关键指标
        assert 'fps' in stats
        assert 'latency' in stats
        assert 'success_rate' in stats
        assert stats['success_rate'] >= 0.95  # 95% 成功率
        
        # 停止监控
        sender.stop_monitoring()

    def teardown_system(self, components):
        """清理系统组件"""
        try:
            # 关闭连接
            if 'socket' in components:
                components['socket'].disconnect()
            
            # 释放资源
            if 'pose' in components:
                components['pose'].close()
            
            # 清理缓存
            if 'drawer' in components:
                components['drawer'].clear_cache()
            
            # 停止所有后台任务
            if 'sender' in components:
                components['sender'].stop_monitoring()
            
        except Exception as e:
            logging.error(f"Teardown error: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    pytest.main(["-v", __file__]) 
import pytest
import numpy as np
import time
from pose.pose_deformer import PoseDeformer
from pose.pose_binding import PoseBinding
from connect.pose_sender import PoseSender
from connect.socket_manager import SocketManager

class TestPoseSystem:
    @pytest.fixture
    def setup_system(self):
        """初始化完整系统"""
        socket_manager = SocketManager()
        sender = PoseSender(socket_manager)
        binding = PoseBinding()
        deformer = PoseDeformer()
        return {
            'socket_manager': socket_manager,
            'sender': sender,
            'binding': binding,
            'deformer': deformer
        }

    def test_end_to_end_processing(self, setup_system):
        """测试端到端处理流程"""
        # 创建测试数据
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pose = self._create_test_pose()
        
        # 1. 创建绑定
        binding = setup_system['binding'].create_binding(frame, pose)
        assert binding.valid
        
        # 2. 执行变形
        deformed = setup_system['deformer'].deform_frame(frame, pose)
        assert deformed is not None
        
        # 3. 发送结果
        success = setup_system['sender'].send_pose_data(
            room="test_room",
            pose_results={
                'frame': deformed,
                'pose': pose,
                'binding': binding
            }
        )
        assert success

    def test_system_stability(self, setup_system):
        """测试系统稳定性"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 运行系统5分钟
        end_time = time.time() + 300
        frame_count = 0
        error_count = 0
        
        while time.time() < end_time:
            try:
                pose = self._create_test_pose(angle=frame_count % 360)
                binding = setup_system['binding'].create_binding(frame, pose)
                deformed = setup_system['deformer'].deform_frame(frame, pose)
                setup_system['sender'].send_pose_data(
                    room="test_room",
                    pose_results={
                        'frame': deformed,
                        'pose': pose,
                        'binding': binding
                    }
                )
                frame_count += 1
            except Exception:
                error_count += 1
            
            time.sleep(1/30)  # 30fps
        
        # 验证系统稳定性
        assert frame_count > 0
        assert error_count / frame_count < 0.01  # 错误率小于1% 
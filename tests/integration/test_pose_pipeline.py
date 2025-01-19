import pytest
from pose.pose_binding import PoseBinding
from connect.pose_sender import PoseSender
from connect.socket_manager import SocketManager

class TestPosePipeline:
    @pytest.fixture
    def setup_pipeline(self):
        """设置测试管道"""
        socket_manager = SocketManager()
        sender = PoseSender(socket_manager)
        binding = PoseBinding()
        return binding, sender
    
    def test_end_to_end_processing(self, setup_pipeline):
        """测试端到端处理流程"""
        binding, sender = setup_pipeline
        
        # 创建测试数据
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pose = self._create_test_pose()
        
        # 处理绑定
        binding_result = binding.create_binding(frame, pose)
        assert binding_result.valid
        
        # 变形处理
        deformed = binding.deform_frame(binding_result, self._create_test_pose(angle=45))
        assert deformed is not None
        
        # 发送结果
        success = sender.send_pose_data(
            room="test_room",
            pose_results={'frame': deformed, 'pose': pose},
            timestamp=time.time()
        )
        assert success

    def test_pipeline_performance(self, setup_pipeline):
        """测试管道性能"""
        binding, sender = setup_pipeline
        
        start_time = time.time()
        for _ in range(100):
            # 执行完整的处理管道
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            pose = self._create_test_pose()
            
            binding_result = binding.create_binding(frame, pose)
            deformed = binding.deform_frame(binding_result, self._create_test_pose(angle=30))
            sender.send_pose_data(
                room="test_room",
                pose_results={'frame': deformed, 'pose': pose},
                timestamp=time.time()
            )
        
        total_time = time.time() - start_time
        assert total_time < 10  # 整个管道处理100帧应在10秒内完成 
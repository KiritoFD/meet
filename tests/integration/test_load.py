import pytest
from integration.test_utils import generate_test_sequence

class TestLoadHandling:
    @pytest.mark.slow
    def test_high_load(self, setup_system):
        """测试高负载情况"""
        frames, poses = generate_test_sequence(frame_count=1000)
        
        success_count = 0
        error_count = 0
        
        for frame, pose in zip(frames, poses):
            try:
                # 处理绑定
                binding = setup_system['binding'].create_binding(frame, pose)
                # 执行变形
                deformed = setup_system['deformer'].deform_frame(frame, pose)
                # 发送数据
                setup_system['sender'].send_pose_data(
                    room="test_room",
                    pose_results={'frame': deformed, 'pose': pose}
                )
                success_count += 1
            except Exception:
                error_count += 1
        
        # 验证成功率
        success_rate = success_count / (success_count + error_count)
        assert success_rate > 0.95  # 期望95%以上的成功率 
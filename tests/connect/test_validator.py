import pytest
from connect.validator import DataValidator
from connect.errors import ValidationError

class TestDataValidator:
    @pytest.fixture
    def setup_validator(self):
        """初始化数据验证器"""
        return DataValidator()

    def test_pose_data_validation(self, setup_validator):
        """测试姿态数据验证"""
        # 有效数据
        valid_pose = {
            'landmarks': [
                {
                    'x': 0.5,
                    'y': 0.5,
                    'z': 0.0,
                    'visibility': 1.0
                }
            ],
            'timestamp': 1234567890.123,
            'room_id': 'test_room'
        }
        assert setup_validator.validate_pose(valid_pose)

        # 无效数据测试
        invalid_cases = [
            # 缺少必要字段
            {'landmarks': []},
            # 坐标值无效
            {'landmarks': [{'x': 'invalid', 'y': 0.5, 'z': 0.0}]},
            # 坐标超出范围
            {'landmarks': [{'x': 2.0, 'y': 0.5, 'z': 0.0}]},
            # 缺少可见度
            {'landmarks': [{'x': 0.5, 'y': 0.5, 'z': 0.0}]},
        ]

        for invalid_data in invalid_cases:
            with pytest.raises(ValidationError):
                setup_validator.validate_pose(invalid_data)

    def test_audio_data_validation(self, setup_validator):
        """测试音频数据验证"""
        # 有效数据
        valid_audio = {
            'data': b'audio_data',
            'sample_rate': 44100,
            'timestamp': 1234567890.123,
            'room_id': 'test_room'
        }
        assert setup_validator.validate_audio(valid_audio)

        # 无效数据测试
        invalid_cases = [
            # 缺少数据
            {'sample_rate': 44100},
            # 无效采样率
            {'data': b'audio', 'sample_rate': -1},
            # 数据类型错误
            {'data': 'not_bytes', 'sample_rate': 44100},
        ]

        for invalid_data in invalid_cases:
            with pytest.raises(ValidationError):
                setup_validator.validate_audio(invalid_data)

    def test_custom_validation_rules(self, setup_validator):
        """测试自定义验证规则"""
        # 添加自定义规则
        def validate_room_name(room_id: str) -> bool:
            return len(room_id) >= 3 and room_id.isalnum()

        setup_validator.add_rule('room_name', validate_room_name)

        # 测试自定义规则
        valid_data = {'room_id': 'room123'}
        invalid_data = {'room_id': 'r!'}

        assert setup_validator.validate_rule('room_name', valid_data['room_id'])
        with pytest.raises(ValidationError):
            setup_validator.validate_rule('room_name', invalid_data['room_id'])

    def test_validation_performance(self, setup_validator):
        """测试验证性能"""
        # 创建大量测试数据
        test_data = {
            'landmarks': [
                {
                    'x': 0.5,
                    'y': 0.5,
                    'z': 0.0,
                    'visibility': 1.0
                }
                for _ in range(1000)  # 1000个关键点
            ],
            'timestamp': 1234567890.123,
            'room_id': 'test_room'
        }

        # 测试验证性能
        import time
        start_time = time.time()
        for _ in range(100):  # 验证100次
            setup_validator.validate_pose(test_data)
        duration = time.time() - start_time

        # 验证应该在1秒内完成
        assert duration < 1.0

    def test_validation_cache(self, setup_validator):
        """测试验证缓存"""
        test_data = {
            'landmarks': [
                {
                    'x': 0.5,
                    'y': 0.5,
                    'z': 0.0,
                    'visibility': 1.0
                }
            ],
            'timestamp': 1234567890.123,
            'room_id': 'test_room'
        }

        # 首次验证
        setup_validator.enable_cache()
        setup_validator.validate_pose(test_data)

        # 再次验证应该使用缓存
        start_time = time.time()
        setup_validator.validate_pose(test_data)
        cache_duration = time.time() - start_time

        # 缓存验证应该更快
        assert cache_duration < 0.001  # 小于1ms

    def test_error_messages(self, setup_validator):
        """测试错误消息"""
        invalid_data = {
            'landmarks': [
                {
                    'x': 2.0,  # 超出范围
                    'y': 0.5,
                    'z': 0.0,
                    'visibility': 1.0
                }
            ]
        }

        try:
            setup_validator.validate_pose(invalid_data)
        except ValidationError as e:
            assert 'x coordinate out of range' in str(e)
            assert e.field == 'landmarks[0].x'
            assert e.code == 'RANGE_ERROR'

    def test_batch_validation(self, setup_validator):
        """测试批量验证"""
        valid_poses = [
            {
                'landmarks': [{'x': 0.5, 'y': 0.5, 'z': 0.0, 'visibility': 1.0}],
                'timestamp': 1234567890.123,
                'room_id': 'test_room'
            }
            for _ in range(10)
        ]

        # 所有数据有效
        results = setup_validator.validate_batch(valid_poses, 'pose')
        assert all(results)

        # 混合有效和无效数据
        invalid_pose = {'landmarks': [{'x': 2.0}]}  # 无效数据
        mixed_poses = valid_poses + [invalid_pose]
        results = setup_validator.validate_batch(mixed_poses, 'pose')
        assert sum(results) == len(valid_poses) 
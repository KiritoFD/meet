import pytest
import numpy as np
import cv2
import logging
from pose.pose_binding import PoseBinding, BindingConfig
from pose.pose_data import PoseData, DeformRegion, BindingPoint
import time

logger = logging.getLogger(__name__)

class TestPoseBinding:
    @pytest.fixture
    def setup_binding(self):
        """初始化测试环境"""
        config = BindingConfig(
            smoothing_factor=0.5,
            min_confidence=0.3,
            joint_limits={
                'shoulder': (-90, 90),
                'elbow': (0, 145),
                'knee': (0, 160)
            }
        )
        return PoseBinding(config)

    @pytest.fixture
    def mock_frame(self):
        """创建测试用图像帧"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # 添加一些特征以便于观察变形效果
        cv2.circle(frame, (320, 240), 50, (255, 255, 255), -1)
        return frame

    @pytest.fixture
    def mock_pose_data(self):
        """创建完整的测试姿态数据"""
        landmarks = []
        # 创建所有必需的关键点
        keypoint_positions = {
            # 躯干关键点
            11: (0.4, 0.3),  # 左肩
            12: (0.6, 0.3),  # 右肩
            23: (0.4, 0.6),  # 左髋
            24: (0.6, 0.6),  # 右髋
            
            # 手臂关键点
            13: (0.3, 0.4),  # 左肘
            14: (0.7, 0.4),  # 右肘
            15: (0.2, 0.5),  # 左腕
            16: (0.8, 0.5),  # 右腕
            
            # 腿部关键点
            25: (0.35, 0.8),  # 左膝
            26: (0.65, 0.8),  # 右膝
            27: (0.3, 0.95),  # 左踝
            28: (0.7, 0.95),  # 右踝
            
            # 面部关键点
            10: (0.5, 0.1),   # 面部轮廓
            70: (0.45, 0.15), # 左眉
            336: (0.55, 0.15), # 右眉
            33: (0.45, 0.2),  # 左眼
            362: (0.55, 0.2), # 右眼
            168: (0.5, 0.25), # 鼻子
            0: (0.5, 0.3),    # 嘴部
        }
        
        # 创建468个关键点(MediaPipe Face Mesh的标准数量)
        for i in range(468):
            if i in keypoint_positions:
                x, y = keypoint_positions[i]
                visibility = 0.9
            else:
                x, y = 0.5, 0.5
                visibility = 0.1
                
            landmarks.append({
                'x': x,
                'y': y,
                'z': 0.0,
                'visibility': visibility
            })
            
        return PoseData(
            landmarks=landmarks,
            timestamp=time.time(),  # 添加时间戳
            confidence=0.9  # 添加置信度
        )

    def test_initialization(self, setup_binding):
        """测试初始化"""
        assert setup_binding.config is not None
        assert setup_binding.region_configs is not None
        assert len(setup_binding.region_configs) > 0
        assert setup_binding._last_valid_binding is None

    def test_required_regions(self, setup_binding, mock_frame, mock_pose_data):
        """测试必需区域的创建"""
        regions = setup_binding.create_binding(mock_frame, mock_pose_data)
        region_names = [r.name for r in regions]
        
        # 验证必需区域存在
        assert 'torso' in region_names
        
        # 获取躯干区域
        torso_region = next(r for r in regions if r.name == 'torso')
        
        # 验证躯干区域的属性
        assert len(torso_region.binding_points) >= 3
        assert torso_region.mask.shape == (480, 640)
        assert np.any(torso_region.mask > 0)

    def test_limb_regions(self, setup_binding, mock_frame, mock_pose_data):
        """测试肢体区域的创建"""
        regions = setup_binding.create_binding(mock_frame, mock_pose_data)
        region_names = [r.name for r in regions]
        
        # 验证肢体区域
        limb_regions = [
            'left_upper_arm', 'left_lower_arm',
            'right_upper_arm', 'right_lower_arm',
            'left_upper_leg', 'left_lower_leg',
            'right_upper_leg', 'right_lower_leg'
        ]
        
        for region_name in limb_regions:
            if region_name in region_names:
                region = next(r for r in regions if r.name == region_name)
                assert len(region.binding_points) >= 2
                assert region.mask is not None

    def test_face_regions(self, setup_binding, mock_frame, mock_pose_data):
        """测试面部区域的创建"""
        regions = setup_binding.create_binding(mock_frame, mock_pose_data)
        region_names = [r.name for r in regions]
        
        # 检查面部区域
        face_regions = [
            'face_contour', 'left_eyebrow', 'right_eyebrow',
            'left_eye', 'right_eye', 'nose', 'mouth'
        ]
        
        for region_name in face_regions:
            if region_name in region_names:
                region = next(r for r in regions if r.name == region_name)
                config = setup_binding.region_configs[region_name]
                assert len(region.binding_points) >= config['min_points']

    def test_region_masks(self, setup_binding, mock_frame, mock_pose_data):
        """测试区域蒙版生成"""
        regions = setup_binding.create_binding(mock_frame, mock_pose_data)
        
        for region in regions:
            mask = region.mask
            # 验证蒙版属性
            assert mask.dtype == np.uint8
            assert mask.shape == (480, 640)
            assert np.min(mask) >= 0
            assert np.max(mask) <= 255
            
            # 验证边缘平滑
            edges = cv2.Canny(mask, 100, 200)
            assert np.sum(edges > 0) > 0

    def test_weight_calculation(self, setup_binding, mock_frame, mock_pose_data):
        """测试权重计算
        
        测试不同区域类型的权重计算是否符合要求：
        1. 躯干区域：权重必须是0.4或0.6
        2. 肢体区域：权重必须是0.3或0.7
        3. 面部轮廓：权重必须是0.5
        4. 其他面部特征：权重必须是0.8
        """
        # 创建区域绑定
        regions = setup_binding.create_binding(mock_frame, mock_pose_data)
        assert len(regions) > 0, "应该至少创建一个区域"
        
        for region in regions:
            weights = [bp.weight for bp in region.binding_points]
            assert len(weights) > 0, f"{region.name}区域应该有权重值"
            
            # 根据区域类型验证权重值
            if region.name == 'torso':
                invalid_weights = [w for w in weights if w not in (0.4, 0.6)]
                assert len(invalid_weights) == 0, \
                    f"躯干区域的权重必须是0.4或0.6，发现无效值：{invalid_weights}"
                    
            elif region.name.endswith(('_arm', '_leg')):
                invalid_weights = [w for w in weights if w not in (0.3, 0.7)]
                assert len(invalid_weights) == 0, \
                    f"肢体区域的权重必须是0.3或0.7，发现无效值：{invalid_weights}"
                    
            elif region.name == 'face_contour':
                invalid_weights = [w for w in weights if not np.isclose(w, 0.5, rtol=1e-5)]
                assert len(invalid_weights) == 0, \
                    f"面部轮廓的权重必须是0.5，发现无效值：{invalid_weights}"
                    
            else:  # 其他面部特征
                invalid_weights = [w for w in weights if not np.isclose(w, 0.8, rtol=1e-5)]
                assert len(invalid_weights) == 0, \
                    f"面部特征的权重必须是0.8，发现无效值：{invalid_weights}"
                    
        # 测试特殊情况
        # 1. 空点列表
        weights = setup_binding._calculate_weights([], 'torso')
        assert len(weights) == 0, "空点列表应该返回空权重列表"
        
        # 2. 单点
        weights = setup_binding._calculate_weights([np.array([0, 0])], 'torso')
        assert len(weights) == 1, "单点应该只有一个权重"
        assert weights[0] == 0.4, "单点躯干区域应该使用0.4权重"
        
        # 3. 超出标准点数的情况
        many_points = [np.array([0, 0])] * 6  # 6个点
        weights = setup_binding._calculate_weights(many_points, 'torso')
        assert len(weights) == 6, "应该为每个点分配权重"
        assert weights[:4] == [0.4, 0.6, 0.6, 0.4], "前4个权重应该保持标准模式"
        assert all(w == 0.4 for w in weights[4:]), "额外的点应该使用0.4权重"

    def test_error_handling(self, setup_binding, mock_frame):
        """测试错误处理"""
        # 测试空输入
        with pytest.raises(ValueError):
            setup_binding.create_binding(None, None)
        
        # 测试无效帧
        with pytest.raises(ValueError):
            setup_binding.create_binding(np.array([]), mock_frame)
        
        # 测试无关键点
        empty_pose = PoseData(
            landmarks=[],
            timestamp=time.time(),
            confidence=0.0
        )
        regions = setup_binding.create_binding(mock_frame, empty_pose)
        assert len(regions) == 0
        
        # 测试低可见度
        low_visibility_pose = PoseData(
            landmarks=[{
                'x': 0.5, 'y': 0.5, 'z': 0.0, 'visibility': 0.1
            }] * 468,
            timestamp=time.time(),
            confidence=0.1
        )
        regions = setup_binding.create_binding(mock_frame, low_visibility_pose)
        assert len(regions) == 0

    def test_cache_mechanism(self, setup_binding, mock_frame, mock_pose_data):
        """测试缓存机制"""
        # 首次创建有效绑定
        initial_regions = setup_binding.create_binding(mock_frame, mock_pose_data)
        assert len(initial_regions) > 0
        
        # 使用无效数据，应返回缓存的结果
        invalid_pose = PoseData(
            landmarks=[],
            timestamp=time.time(),
            confidence=0.0
        )
        fallback_regions = setup_binding.create_binding(mock_frame, invalid_pose)
        assert len(fallback_regions) > 0
        assert len(fallback_regions) == len(initial_regions)

    def test_performance(self, setup_binding, mock_frame, mock_pose_data):
        """测试性能"""
        import time
        
        # 测试绑定创建性能
        start_time = time.time()
        regions = setup_binding.create_binding(mock_frame, mock_pose_data)
        end_time = time.time()
        
        assert end_time - start_time < 0.01  # 10ms限制
        
        # 测试内存使用
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        assert memory_info.rss / (1024 * 1024) < 100  # 100MB限制

    @staticmethod
    def _create_test_pose(angle: float = 0, visibility: float = 1.0) -> PoseData:
        """创建测试姿态数据"""
        landmarks = []
        for i in range(468):  # MediaPipe Face Mesh的标准数量
            x = 0.5 + 0.1 * np.cos(np.radians(angle))
            y = 0.5 + 0.1 * np.sin(np.radians(angle))
            landmarks.append({
                'x': x,
                'y': y,
                'z': 0.0,
                'visibility': visibility
            })
        return PoseData(
            landmarks=landmarks,
            timestamp=time.time(),
            confidence=visibility
        )

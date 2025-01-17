import pytest
import numpy as np
import cv2
from pose.pose_deformer import PoseDeformer
from pose.pose_data import PoseData

class TestPoseDeformer:
    @pytest.fixture
    def setup_deformer(self):
        """初始化测试环境"""
        return PoseDeformer()

    def test_basic_deformation(self, setup_deformer):
        """测试基本变形"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, (320, 240), 50, (255, 255, 255), -1)
        pose = self._create_test_pose()
        
        deformed = setup_deformer.deform_frame(frame, pose)
        assert deformed is not None
        assert deformed.shape == frame.shape

    def test_pose_interpolation(self, setup_deformer):
        """测试姿态插值"""
        # 创建两个测试姿态
        pose1 = self._create_test_pose(angle=0)
        pose2 = self._create_test_pose(angle=90)
        
        # 测试不同插值比例
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            interpolated = setup_deformer.interpolate(pose1, pose2, t)
            assert len(interpolated.landmarks) == len(pose1.landmarks)
            # 验证插值结果在两个姿态之间
            for i in range(len(interpolated.landmarks)):
                assert interpolated.landmarks[i]['x'] >= min(pose1.landmarks[i]['x'], pose2.landmarks[i]['x'])
                assert interpolated.landmarks[i]['x'] <= max(pose1.landmarks[i]['x'], pose2.landmarks[i]['x'])

    def test_sequence_smoothing(self, setup_deformer):
        """测试序列平滑"""
        # 创建带噪声的姿态序列
        poses = [
            self._create_test_pose(angle=i + np.random.normal(0, 5))
            for i in range(0, 90, 10)
        ]
        
        smoothed = setup_deformer.smooth_sequence(poses)
        assert len(smoothed) == len(poses)
        
        # 验证平滑效果
        for i in range(1, len(smoothed)-1):
            curr = smoothed[i].landmarks[0]['x']
            prev = smoothed[i-1].landmarks[0]['x']
            next_val = smoothed[i+1].landmarks[0]['x']
            # 验证当前值在前后值之间
            assert min(prev, next_val) <= curr <= max(prev, next_val)

    def test_pose_prediction(self, setup_deformer):
        """测试姿态预测"""
        # 创建历史姿态序列
        history = [
            self._create_test_pose(angle=i * 10)
            for i in range(5)
        ]
        
        predicted = setup_deformer.predict_next(history)
        assert predicted is not None
        # 验证预测趋势
        last_angle = np.arctan2(history[-1].landmarks[0]['y'], 
                              history[-1].landmarks[0]['x'])
        pred_angle = np.arctan2(predicted.landmarks[0]['y'], 
                              predicted.landmarks[0]['x'])
        assert pred_angle > last_angle

    def test_error_handling(self, setup_deformer):
        """测试错误处理"""
        # 测试序列太短
        with pytest.raises(ValueError):
            setup_deformer.smooth_sequence([self._create_test_pose()])
        
        # 测试无效插值参数
        pose1 = self._create_test_pose()
        pose2 = self._create_test_pose()
        with pytest.raises(ValueError):
            setup_deformer.interpolate(pose1, pose2, 1.5)
        
        # 测试无效预测输入
        with pytest.raises(ValueError):
            setup_deformer.predict_next([])

    def test_large_deformation(self, setup_deformer):
        """测试大幅度变形"""
        poses = [
            self._create_test_pose(angle=0),
            self._create_test_pose(angle=180)
        ]
        
        # 测试大角度插值
        for t in np.linspace(0, 1, 10):
            interpolated = setup_deformer.interpolate(poses[0], poses[1], t)
            assert not np.any(np.isnan([lm['x'] for lm in interpolated.landmarks]))
            assert not np.any(np.isnan([lm['y'] for lm in interpolated.landmarks]))

    def test_smoothing_window(self, setup_deformer):
        """测试不同平滑窗口大小"""
        poses = [self._create_test_pose(angle=i) for i in range(10)]
        
        # 测试不同窗口大小
        for window in [3, 5, 7]:
            deformer = PoseDeformer(smoothing_window=window)
            smoothed = deformer.smooth_sequence(poses)
            assert len(smoothed) == len(poses)

    def test_real_time_deformation(self, setup_deformer):
        """测试实时变形性能"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, (320, 240), 50, (255, 255, 255), -1)
        
        # 模拟实时姿态流
        poses = [self._create_test_pose(angle=i) for i in range(0, 360, 10)]
        
        import time
        start_time = time.time()
        for pose in poses:
            deformed = setup_deformer.deform_frame(frame, pose)
            assert deformed is not None
        
        avg_time = (time.time() - start_time) / len(poses)
        assert avg_time < 0.033  # 确保每帧处理时间小于33ms（30fps）

    def test_feature_preservation(self, setup_deformer):
        """测试特征保持"""
        # 创建带有多个特征的测试帧
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        features = [
            {'pos': (320, 240), 'color': (255, 0, 0)},
            {'pos': (160, 120), 'color': (0, 255, 0)},
            {'pos': (480, 360), 'color': (0, 0, 255)}
        ]
        
        for feature in features:
            cv2.circle(frame, feature['pos'], 20, feature['color'], -1)
        
        pose = self._create_test_pose(angle=45)
        deformed = setup_deformer.deform_frame(frame, pose)
        
        # 验证特征颜色和形状保持
        for feature in features:
            # 在变形后的图像中查找对应颜色的区域
            color_mask = np.all(deformed == feature['color'], axis=2)
            assert np.any(color_mask)  # 确保颜色存在
            # 验证形状完整性（连通区域）
            from scipy import ndimage
            labeled, num = ndimage.label(color_mask)
            assert num > 0  # 至少有一个连通区域

    def test_motion_continuity(self, setup_deformer):
        """测试运动连续性"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, (320, 240), 50, (255, 255, 255), -1)
        
        # 创建连续运动序列
        motions = [
            {'angle': 0, 'scale': 1.0},
            {'angle': 30, 'scale': 1.2},
            {'angle': 60, 'scale': 0.8},
            {'angle': 90, 'scale': 1.0}
        ]
        
        prev_centroids = None
        for motion in motions:
            pose = self._create_test_pose(**motion)
            deformed = setup_deformer.deform_frame(frame, pose)
            
            # 计算变形后白色区域的质心
            white_pixels = np.where(deformed[:,:,0] > 128)
            if len(white_pixels[0]) > 0:
                centroids = (np.mean(white_pixels[1]), np.mean(white_pixels[0]))
                
                if prev_centroids is not None:
                    # 计算质心移动距离
                    dist = np.sqrt((centroids[0] - prev_centroids[0])**2 + 
                                 (centroids[1] - prev_centroids[1])**2)
                    # 验证移动平滑
                    assert dist < 100  # 根据实际情况调整阈值
                
                prev_centroids = centroids

    @staticmethod
    def _create_test_pose(angle: float = 0.0):
        """创建测试姿态数据"""
        return PoseData(landmarks=[{
            'x': np.cos(np.radians(angle)),
            'y': np.sin(np.radians(angle)),
            'z': 0.0,
            'visibility': 1.0
        } for _ in range(33)])  # MediaPipe姿态点数量 
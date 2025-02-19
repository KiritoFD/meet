import pytest
from unittest.mock import Mock, patch
import numpy as np
import cv2
import sys
import os
import time

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from pose.types import PoseData, Landmark, DeformRegion
from run import capture_reference, app, process_frame  # 导入 app 和 process_frame

@pytest.fixture
def mock_camera_manager():
    manager = Mock()
    
    def read_frame_effect():
        """模拟读取摄像头画面的逻辑"""
        if not manager.is_running:
            return None  # 摄像头未运行时返回 None
        return np.ones((480, 640, 3), dtype=np.uint8)  # 正常情况返回非零数组
        
    manager.is_running = True
    manager.read_frame = Mock(side_effect=read_frame_effect)
    return manager

@pytest.fixture
def mock_pose():
    pose = Mock()
    
    def process_effect(image):
        """模拟姿态检测的逻辑"""
        if image is None or not image.any():
            return Mock(pose_landmarks=None)
            
        # 创建姿态检测结果
        landmarks = []
        for i in range(33):
            lm = Mock()
            lm.x = 0.5
            lm.y = 0.5
            lm.z = 0.0
            lm.visibility = 0.9
            landmarks.append(lm)
            
        results = Mock()
        results.pose_landmarks = Mock()
        results.pose_landmarks.landmark = landmarks
        return results
        
    pose.process = Mock(side_effect=process_effect)
    return pose

@pytest.fixture
def mock_face_mesh():
    face_mesh = Mock()
    
    def process_effect(image):
        """模拟面部检测的逻辑"""
        if image is None or not image.any():
            return Mock(multi_face_landmarks=None)
            
        # 创建面部检测结果
        landmarks = []
        for i in range(468):
            lm = Mock()
            lm.x = 0.5
            lm.y = 0.5
            lm.z = 0.0
            landmarks.append(lm)
            
        face_landmarks = Mock()
        face_landmarks.landmark = landmarks
        results = Mock()
        results.multi_face_landmarks = [face_landmarks]
        return results
        
    face_mesh.process = Mock(side_effect=process_effect)
    return face_mesh

def create_multi_regions(landmarks, image_shape):
    """创建多个细致的变形区域"""
    h, w = image_shape[:2] if image_shape else (480, 640)
    regions = []
    
    # 创建基础遮罩
    base_mask = np.ones((h, w), dtype=np.uint8)
    
    # 面部区域 - 细分为多个子区域
    face_regions = [
        # 额头区域
        DeformRegion(
            name='forehead',
            type='face',
            center=(320, 50),
            binding_points=[
                Mock(landmark_index=10, local_coords=(0, 0)),
                Mock(landmark_index=151, local_coords=(0.1, 0)),
                Mock(landmark_index=9, local_coords=(-0.1, 0))
            ],
            mask=base_mask.copy()
        ),
        # 左眉毛
        DeformRegion(
            name='left_eyebrow',
            type='face',
            center=(300, 70),
            binding_points=[
                Mock(landmark_index=282, local_coords=(0, 0)),
                Mock(landmark_index=295, local_coords=(0.05, 0)),
                Mock(landmark_index=276, local_coords=(-0.05, 0))
            ],
            mask=base_mask.copy()
        ),
        # 右眉毛
        DeformRegion(
            name='right_eyebrow',
            type='face',
            center=(340, 70),
            binding_points=[
                Mock(landmark_index=52, local_coords=(0, 0)),
                Mock(landmark_index=65, local_coords=(0.05, 0)),
                Mock(landmark_index=46, local_coords=(-0.05, 0))
            ],
            mask=base_mask.copy()
        ),
        # 左眼
        DeformRegion(
            name='left_eye',
            type='face',
            center=(300, 90),
            binding_points=[
                Mock(landmark_index=362, local_coords=(0, 0)),
                Mock(landmark_index=374, local_coords=(0.03, 0)),
                Mock(landmark_index=386, local_coords=(0, 0.03))
            ],
            mask=base_mask.copy()
        ),
        # 右眼
        DeformRegion(
            name='right_eye',
            type='face',
            center=(340, 90),
            binding_points=[
                Mock(landmark_index=33, local_coords=(0, 0)),
                Mock(landmark_index=246, local_coords=(0.03, 0)),
                Mock(landmark_index=161, local_coords=(0, 0.03))
            ],
            mask=base_mask.copy()
        ),
        # 鼻子
        DeformRegion(
            name='nose',
            type='face',
            center=(320, 110),
            binding_points=[
                Mock(landmark_index=1, local_coords=(0, 0)),
                Mock(landmark_index=4, local_coords=(0.02, 0.02)),
                Mock(landmark_index=5, local_coords=(-0.02, 0.02))
            ],
            mask=base_mask.copy()
        ),
        # 左脸颊
        DeformRegion(
            name='left_cheek',
            type='face',
            center=(290, 120),
            binding_points=[
                Mock(landmark_index=425, local_coords=(0, 0)),
                Mock(landmark_index=427, local_coords=(0.03, 0)),
                Mock(landmark_index=429, local_coords=(0, 0.03))
            ],
            mask=base_mask.copy()
        ),
        # 右脸颊
        DeformRegion(
            name='right_cheek',
            type='face',
            center=(350, 120),
            binding_points=[
                Mock(landmark_index=205, local_coords=(0, 0)),
                Mock(landmark_index=207, local_coords=(0.03, 0)),
                Mock(landmark_index=209, local_coords=(0, 0.03))
            ],
            mask=base_mask.copy()
        ),
        # 嘴巴
        DeformRegion(
            name='mouth',
            type='face',
            center=(320, 140),
            binding_points=[
                Mock(landmark_index=0, local_coords=(0, 0)),
                Mock(landmark_index=17, local_coords=(0.03, 0)),
                Mock(landmark_index=267, local_coords=(-0.03, 0))
            ],
            mask=base_mask.copy()
        ),
        # 下巴
        DeformRegion(
            name='chin',
            type='face',
            center=(320, 160),
            binding_points=[
                Mock(landmark_index=152, local_coords=(0, 0)),
                Mock(landmark_index=148, local_coords=(0.02, 0)),
                Mock(landmark_index=149, local_coords=(-0.02, 0))
            ],
            mask=base_mask.copy()
        )
    ]
    
    # 添加测试验证点
    print("\n=== 面部区域创建 ===")
    print(f"创建了 {len(face_regions)} 个面部区域")
    for region in face_regions:
        print(f"区域 {region.name}: {len(region.binding_points)} 个绑定点")
    
    regions.extend(face_regions)
    
    # 身体区域 - 添加更多细节
    body_regions = [
        # 头部整体
        DeformRegion(
            name='head',
            type='body',
            center=(320, 120),
            binding_points=[
                Mock(landmark_index=0, local_coords=(0, 0)),
                Mock(landmark_index=11, local_coords=(-0.3, 0.5)),
                Mock(landmark_index=12, local_coords=(0.3, 0.5))
            ],
            mask=base_mask.copy()
        ),
        # 颈部
        DeformRegion(
            name='neck',
            type='body',
            center=(320, 180),
            binding_points=[
                Mock(landmark_index=0, local_coords=(0, -0.2)),
                Mock(landmark_index=11, local_coords=(-0.1, 0)),
                Mock(landmark_index=12, local_coords=(0.1, 0))
            ],
            mask=base_mask.copy()
        ),
        # 左肩
        DeformRegion(
            name='left_shoulder',
            type='body',
            center=(270, 200),
            binding_points=[
                Mock(landmark_index=11, local_coords=(0, 0)),
                Mock(landmark_index=13, local_coords=(-0.2, 0)),
                Mock(landmark_index=23, local_coords=(0, 0.2))
            ],
            mask=base_mask.copy()
        ),
        # 右肩
        DeformRegion(
            name='right_shoulder',
            type='body',
            center=(370, 200),
            binding_points=[
                Mock(landmark_index=12, local_coords=(0, 0)),
                Mock(landmark_index=14, local_coords=(0.1, 0)),
                Mock(landmark_index=24, local_coords=(0, 0.1))
            ],
            mask=base_mask.copy()
        ),
        # 上胸部
        DeformRegion(
            name='upper_chest',
            type='body',
            center=(320, 220),
            binding_points=[
                Mock(landmark_index=11, local_coords=(-0.2, 0)),
                Mock(landmark_index=12, local_coords=(0.2, 0)),
                Mock(landmark_index=23, local_coords=(0, 0.2))
            ],
            mask=base_mask.copy()
        ),
        # 左上臂
        DeformRegion(
            name='left_upper_arm',
            type='limb',
            center=(250, 240),
            binding_points=[
                Mock(landmark_index=11, local_coords=(0, 0)),
                Mock(landmark_index=13, local_coords=(0, 0.15)),
                Mock(landmark_index=15, local_coords=(0, 0.3))
            ],
            mask=base_mask.copy()
        ),
        # 右上臂
        DeformRegion(
            name='right_upper_arm',
            type='limb',
            center=(390, 240),
            binding_points=[
                Mock(landmark_index=12, local_coords=(0, 0)),
                Mock(landmark_index=14, local_coords=(0, 0.15)),
                Mock(landmark_index=16, local_coords=(0, 0.3))
            ],
            mask=base_mask.copy()
        ),
        # 左前臂
        DeformRegion(
            name='left_forearm',
            type='limb',
            center=(240, 280),
            binding_points=[
                Mock(landmark_index=13, local_coords=(0, 0)),
                Mock(landmark_index=15, local_coords=(0, 0.2))
            ],
            mask=base_mask.copy()
        ),
        # 右前臂
        DeformRegion(
            name='right_forearm',
            type='limb',
            center=(400, 280),
            binding_points=[
                Mock(landmark_index=14, local_coords=(0, 0)),
                Mock(landmark_index=16, local_coords=(0, 0.2))
            ],
            mask=base_mask.copy()
        )
    ]
    
    print("\n=== 身体区域创建 ===")
    print(f"创建了 {len(body_regions)} 个身体区域")
    for region in body_regions:
        print(f"区域 {region.name}: {len(region.binding_points)} 个绑定点")
    
    regions.extend(body_regions)
    
    # 添加区域间关系验证
    print("\n=== 区域关系验证 ===")
    for r1 in regions:
        for r2 in regions:
            if r1 != r2 and r1.type == r2.type:
                dist = np.linalg.norm(np.array(r1.center) - np.array(r2.center))
                print(f"{r1.name} 和 {r2.name} 的距离: {dist:.2f}像素")
    
    return regions

@pytest.fixture
def mock_pose_binding():
    binding = Mock()
    binding.create_binding = Mock(side_effect=create_multi_regions)
    return binding

@pytest.fixture
def mock_frame():
    """创建一个测试用的图像帧"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # 添加一些简单的图案以便于识别
    cv2.rectangle(frame, (200, 150), (440, 330), (255, 255, 255), -1)
    return frame

@pytest.fixture
def mock_pose_data():
    """创建一个完整的姿态数据对象"""
    landmarks = []
    for i in range(33):  # MediaPipe Pose 的标准点数
        x = 0.5
        y = 0.5
        visibility = 0.9
        
        # 设置关键点的特定位置
        if i == 0:  # 鼻子
            x, y = 0.5, 0.2
        elif i in [11, 12]:  # 肩膀
            x = 0.3 if i == 11 else 0.7
            y = 0.3
        
        landmarks.append(Landmark(x=x, y=y, z=0.0, visibility=visibility))
    
    return PoseData(
        landmarks=landmarks,
        face_landmarks=[Landmark(x=0.5, y=0.2, z=0.0) for _ in range(468)],
        timestamp=time.time(),
        confidence=0.9
    )

@pytest.fixture
def mock_frame_processor():
    """模拟帧处理器"""
    processor = Mock()
    processor.reference_frame = None
    processor.reference_pose = None
    processor.regions = None
    processor.deformed_frame = None
    return processor

@pytest.fixture
def mock_detector():
    """模拟检测器"""
    detector = Mock()
    
    def process_frame_effect(frame):
        """模拟帧处理"""
        if frame is None:
            return None
            
        # 创建一个包含姿态关键点的检测结果
        class PoseLandmark:
            def __init__(self, x, y, z, visibility):
                self.x = x
                self.y = y
                self.z = z
                self.visibility = visibility
                
        class PoseLandmarks:
            def __init__(self, landmarks):
                self.landmark = landmarks
                
        class DetectionResult:
            def __init__(self, landmarks):
                self.pose_landmarks = landmarks
                
        # 创建向右平移的姿态关键点
        landmarks = []
        for i in range(33):
            # 对于当前帧，所有关键点向右平移 0.2
            lm = PoseLandmark(
                x=0.7,  # 0.5 + 0.2
                y=0.5,
                z=0.0,
                visibility=0.9
            )
            landmarks.append(lm)
            
        pose_landmarks = PoseLandmarks(landmarks)
        results = DetectionResult(pose_landmarks)
        return results
        
    def draw_detections_effect(frame, detection_result):
        """模拟绘制检测结果"""
        # 在帧上绘制关键点位置
        result = frame.copy()
        if result is not None and result.size > 0:
            h, w = result.shape[:2]
            # 绘制所有关键点
            for lm in detection_result.pose_landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(result, (x, y), 5, (0, 255, 0), -1)
        return result
        
    detector.process_frame = Mock(side_effect=process_frame_effect)
    detector.draw_detections = Mock(side_effect=draw_detections_effect)
    return detector

@pytest.fixture
def mock_pose_deformer():
    """模拟变形器"""
    deformer = Mock()
    
    def deform_effect(reference_frame, reference_pose, current_frame, detection_result, regions):
        """模拟变形处理"""
        if current_frame is not None and current_frame.size > 0:
            result = current_frame.copy()
            h, w = result.shape[:2]
            
            # 根据姿态变化创建变形效果
            # 假设姿态向右移动了 0.2，我们将图像内容向右平移
            shift = int(0.2 * w)  # 计算平移像素数
            
            # 使用仿射变换实现平移
            M = np.float32([[1, 0, shift], [0, 1, 0]])
            result = cv2.warpAffine(result, M, (w, h))
            
            # 在变形区域添加一些视觉标记
            for region in regions:
                if isinstance(region, Mock):
                    # 如果是 Mock 对象，使用默认值
                    center = np.array([w//2, h//2])
                else:
                    center = region.center
                center = center.astype(int)
                cv2.circle(result, tuple(center), 10, (0, 0, 255), -1)
                
            return result
        return current_frame
        
    deformer.deform = Mock(side_effect=deform_effect)
    return deformer

def test_capture_reference_success(mock_camera_manager, mock_pose, mock_face_mesh, mock_pose_binding):
    """测试成功捕获参考帧的情况"""
    with app.app_context():
        with patch('run.camera_manager', mock_camera_manager), \
             patch('run.pose', mock_pose), \
             patch('run.face_mesh', mock_face_mesh), \
             patch('run.pose_binding', mock_pose_binding):
            
            response = capture_reference()
            # 如果 response 是元组，获取第一个元素
            if isinstance(response, tuple):
                response = response[0]
            response_data = response.get_json()
            
            assert response.status_code == 200
            assert response_data['success'] is True
            assert 'regions_info' in response_data['details']
            assert response_data['details']['regions_info']['body'] == 1
            assert response_data['details']['regions_info']['face'] == 1

def test_capture_reference_no_camera(mock_camera_manager):
    """测试摄像头未运行的情况"""
    mock_camera_manager.is_running = False
    
    with app.app_context():
        with patch('run.camera_manager', mock_camera_manager):
            response = capture_reference()
            response, status_code = response
            response_data = response.get_json()
            
            assert status_code == 400
            assert response_data['success'] is False
            assert '摄像头未运行' in response_data['message']

def test_capture_reference_no_frame(mock_camera_manager):
    """测试无法获取摄像头画面的情况"""
    mock_camera_manager.is_running = True
    mock_camera_manager.read_frame.return_value = None  # 直接返回 None
    
    with app.app_context():
        with patch('run.camera_manager', mock_camera_manager):
            response = capture_reference()
            response, status_code = response
            response_data = response.get_json()
            
            assert status_code == 500
            assert response_data['success'] is False
            assert '无法获取摄像头画面' in response_data['message']

def test_capture_reference_invalid_frame(mock_camera_manager):
    """测试获取到无效画面的情况"""
    mock_camera_manager.is_running = True
    mock_camera_manager.read_frame.return_value = np.zeros((0, 0, 3), dtype=np.uint8)  # 返回空图像
    
    with app.app_context():
        with patch('run.camera_manager', mock_camera_manager):
            response = capture_reference()
            response, status_code = response
            response_data = response.get_json()
            
            assert status_code == 500
            assert response_data['success'] is False
            assert '无效的摄像头画面' in response_data['message']

def test_capture_reference_no_pose(mock_camera_manager, mock_pose):
    """测试未检测到人物姿态的情况"""
    def process_no_pose(image):
        """返回无姿态检测结果"""
        return Mock(pose_landmarks=None)
        
    mock_pose.process = Mock(side_effect=process_no_pose)
    
    with app.app_context():
        with patch('run.camera_manager', mock_camera_manager), \
             patch('run.pose', mock_pose):
            response = capture_reference()
            response, status_code = response  # 解包元组
            response_data = response.get_json()
            
            assert status_code == 400
            assert response_data['success'] is False
            assert '未检测到人物姿态' in response_data['message']

def test_capture_reference_with_valid_data(mock_camera_manager, mock_pose, mock_face_mesh, mock_pose_binding, mock_frame, mock_pose_data):
    """测试使用有效数据捕获参考帧"""
    mock_camera_manager.read_frame.return_value = mock_frame
    mock_pose.process.return_value.pose_landmarks.landmark = mock_pose_data.landmarks
    
    with app.app_context():
        with patch('run.camera_manager', mock_camera_manager), \
             patch('run.pose', mock_pose), \
             patch('run.face_mesh', mock_face_mesh), \
             patch('run.pose_binding', mock_pose_binding):
            
            response = capture_reference()
            if isinstance(response, tuple):
                response = response[0]
            response_data = response.get_json()
            
            # 验证基本响应
            assert response.status_code == 200
            assert response_data['success'] is True
            
            # 验证返回的区域信息
            assert 'regions_info' in response_data['details']
            regions_info = response_data['details']['regions_info']
            assert regions_info['body'] == 1
            assert regions_info['face'] == 1
            
            # 验证是否保存了参考帧
            assert 'reference_frame' in response_data['details']
            assert response_data['details']['reference_frame'] is not None

def test_capture_reference_with_low_confidence(mock_camera_manager, mock_pose, mock_face_mesh, mock_pose_binding):
    """测试姿态检测置信度低的情况"""
    def process_with_low_confidence(image):
        """返回低置信度的姿态检测结果"""
        results = Mock()
        landmarks = []
        for i in range(33):
            lm = Mock()
            lm.x = 0.5
            lm.y = 0.5
            lm.z = 0.0
            lm.visibility = 0.3  # 低可见度
            landmarks.append(lm)
        results.pose_landmarks = Mock()
        results.pose_landmarks.landmark = landmarks
        return results
        
    mock_pose.process = Mock(side_effect=process_with_low_confidence)
    
    with app.app_context():
        with patch('run.camera_manager', mock_camera_manager), \
             patch('run.pose', mock_pose), \
             patch('run.face_mesh', mock_face_mesh), \
             patch('run.frame_processor', Mock()), \
             patch('run.pose_binding', mock_pose_binding):
            
            response = capture_reference()
            response, status_code = response  # 解包元组
            response_data = response.get_json()
            
            assert status_code == 400
            assert response_data['success'] is False
            assert '姿态检测置信度过低' in response_data['message']

def test_capture_reference_with_partial_detection(mock_camera_manager, mock_pose, mock_face_mesh, mock_pose_binding):
    """测试只检测到部分关键点的情况"""
    def process_with_partial_landmarks(image):
        """返回只有部分关键点的检测结果"""
        results = Mock()
        landmarks = []
        for i in range(15):  # 只有15个关键点
            lm = Mock()
            lm.x = 0.5
            lm.y = 0.5
            lm.z = 0.0
            lm.visibility = 0.9
            landmarks.append(lm)
        results.pose_landmarks = Mock()
        results.pose_landmarks.landmark = landmarks
        return results
        
    mock_pose.process = Mock(side_effect=process_with_partial_landmarks)
    
    with app.app_context():
        with patch('run.camera_manager', mock_camera_manager), \
             patch('run.pose', mock_pose), \
             patch('run.face_mesh', mock_face_mesh), \
             patch('run.pose_binding', mock_pose_binding):
            
            response = capture_reference()
            response, status_code = response  # 解包元组
            response_data = response.get_json()
            
            assert status_code == 400
            assert response_data['success'] is False
            assert '检测到的关键点不完整' in response_data['message']

def test_capture_reference_with_face_detection_failure(mock_camera_manager, mock_pose, mock_face_mesh, mock_pose_binding):
    """测试面部检测失败的情况"""
    def process_no_face(image):
        """返回无面部检测结果"""
        return Mock(multi_face_landmarks=None)
        
    mock_face_mesh.process = Mock(side_effect=process_no_face)
    
    with app.app_context():
        with patch('run.camera_manager', mock_camera_manager), \
             patch('run.pose', mock_pose), \
             patch('run.face_mesh', mock_face_mesh), \
             patch('run.frame_processor', Mock()), \
             patch('run.pose_binding', mock_pose_binding):
            
            response = capture_reference()
            response, status_code = response  # 解包元组
            response_data = response.get_json()
            
            assert status_code == 400
            assert response_data['success'] is False
            assert '未检测到面部' in response_data['message']

def test_capture_and_deform_flow(mock_camera_manager, mock_pose, mock_face_mesh, 
                                mock_pose_binding, mock_frame_processor, 
                                mock_detector, mock_pose_deformer):
    """测试完整的捕获和变形流程"""
    # 1. 准备测试数据 - 使用更容易区分的图像
    reference_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255  # 白色参考帧
    cv2.circle(reference_frame, (320, 240), 100, (0, 0, 255), -1)  # 添加红色圆形
    
    current_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # 黑色当前帧
    cv2.circle(current_frame, (320, 240), 100, (0, 255, 0), -1)  # 添加绿色圆形
    
    # 2. 设置参考帧的检测结果
    reference_landmarks = []
    for i in range(33):
        lm = Mock()
        # 设置参考姿态的关键点位置
        if i == 0:  # 鼻子
            lm.x, lm.y = 0.5, 0.2
        elif i == 11:  # 左肩
            lm.x, lm.y = 0.4, 0.3
        elif i == 12:  # 右肩
            lm.x, lm.y = 0.6, 0.3
        else:
            lm.x, lm.y = 0.5, 0.5
        lm.z = 0.0
        lm.visibility = 0.9
        reference_landmarks.append(lm)
    
    # 3. 设置当前帧的检测结果（向右移动）
    current_landmarks = []
    for i in range(33):
        lm = Mock()
        # 设置当前姿态的关键点位置（整体向右移动0.2）
        if i == 0:  # 鼻子
            lm.x, lm.y = 0.7, 0.2
        elif i == 11:  # 左肩
            lm.x, lm.y = 0.6, 0.3
        elif i == 12:  # 右肩
            lm.x, lm.y = 0.8, 0.3
        else:
            lm.x, lm.y = 0.7, 0.5
        lm.z = 0.0
        lm.visibility = 0.9
        current_landmarks.append(lm)
    
    # 4. 设置 mock 对象的行为
    mock_camera_manager.is_running = True
    mock_camera_manager.read_frame.side_effect = [reference_frame, current_frame]
    
    # 设置姿态检测的行为
    def create_pose_result(landmarks):
        result = Mock()
        result.pose_landmarks = Mock()
        result.pose_landmarks.landmark = landmarks
        return result
        
    mock_pose.process.side_effect = [
        create_pose_result(reference_landmarks),
        create_pose_result(current_landmarks)
    ]
    
    # 设置面部检测的行为
    def create_face_result():
        result = Mock()
        face_landmarks = []
        for i in range(468):
            lm = Mock()
            lm.x = 0.5 + (0.2 if i > 233 else 0)  # 第二次调用时向右移动
            lm.y = 0.2
            lm.z = 0.0
            face_landmarks.append(lm)
        result.multi_face_landmarks = [Mock(landmark=face_landmarks)]
        return result
        
    mock_face_mesh.process.side_effect = [create_face_result(), create_face_result()]
    
    # 设置绑定区域的行为
    def create_binding_regions(frame, pose_data):
        h, w = frame.shape[:2]
        regions = [
            DeformRegion(
                name='torso',
                center=np.array([w//2, h//2]),
                binding_points=[
                    Mock(landmark_index=11, local_coords=np.array([-50, 0])),
                    Mock(landmark_index=12, local_coords=np.array([50, 0]))
                ],
                mask=np.ones((h, w), dtype=np.uint8),
                type='body'
            )
        ]
        return regions
        
    mock_pose_binding.create_binding.side_effect = create_binding_regions
    
    with app.app_context():
        with patch('run.camera_manager', mock_camera_manager), \
             patch('run.pose', mock_pose), \
             patch('run.face_mesh', mock_face_mesh), \
             patch('run.pose_binding', mock_pose_binding), \
             patch('run.frame_processor', mock_frame_processor), \
             patch('run.detector', mock_detector), \
             patch('run.pose_deformer', mock_pose_deformer):
            
            # 5. 执行捕获参考帧
            response = capture_reference()
            response, status_code = response
            response_data = response.get_json()
            
            # 验证捕获成功
            assert status_code == 200, "捕获参考帧应该成功"
            assert response_data['success'] is True, "捕获参考帧应该返回成功"
            assert 'regions_info' in response_data['details'], "应该包含变形区域信息"
            
            # 验证参考数据保存
            assert mock_frame_processor.reference_frame is not None, "参考帧应该被保存"
            assert mock_frame_processor.reference_pose is not None, "参考姿态应该被保存"
            assert mock_frame_processor.regions is not None, "变形区域应该被保存"
            
            # 验证参考帧的内容
            assert np.array_equal(mock_frame_processor.reference_frame, reference_frame), \
                "保存的参考帧应该与原始帧相同"
            
            # 6. 处理当前帧
            deformed_frame = process_frame(current_frame)
            
            # 验证变形结果
            assert deformed_frame is not None, "应该返回变形后的帧"
            assert deformed_frame.shape == current_frame.shape, "变形后的帧尺寸应该保持不变"
            assert deformed_frame.dtype == current_frame.dtype, "变形后的帧类型应该保持不变"
            
            # 验证变形效果
            assert np.any(deformed_frame != current_frame), "变形后的帧应该与原始帧不同"
            
            # 计算帧差异的位置和程度
            diff = cv2.absdiff(deformed_frame, current_frame)
            movement = np.mean(diff)
            assert movement > 0, "应该检测到明显的变形效果"
            
            # 验证变形方向
            # 假设向右移动，计算左右两侧的差异
            left_diff = np.mean(diff[:, :320])
            right_diff = np.mean(diff[:, 320:])
            assert right_diff > left_diff, "变形应该向右移动"

def test_multi_region_deform_flow(mock_camera_manager, mock_pose, mock_face_mesh, 
                                mock_pose_binding, mock_frame_processor, 
                                mock_detector, mock_pose_deformer):
    """测试多个变形区域的完整流程"""
    # 1. 准备测试数据
    reference_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
    # 添加多个标记点，代表不同的身体部位
    cv2.circle(reference_frame, (320, 100), 30, (0, 0, 255), -1)  # 头部
    cv2.circle(reference_frame, (320, 200), 50, (255, 0, 0), -1)  # 躯干
    cv2.circle(reference_frame, (270, 200), 20, (0, 255, 0), -1)  # 左肩
    cv2.circle(reference_frame, (370, 200), 20, (0, 255, 0), -1)  # 右肩
    
    current_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # 在当前帧中，所有标记点向右移动50像素
    cv2.circle(current_frame, (370, 100), 30, (0, 0, 255), -1)  # 头部
    cv2.circle(current_frame, (370, 200), 50, (255, 0, 0), -1)  # 躯干
    cv2.circle(current_frame, (320, 200), 20, (0, 255, 0), -1)  # 左肩
    cv2.circle(current_frame, (420, 200), 20, (0, 255, 0), -1)  # 右肩

    # 定义要检查的区域
    regions_to_check = [
        # 面部区域
        ('forehead', (290, 30, 350, 70)),
        ('left_eye', (280, 80, 320, 100)),
        ('right_eye', (320, 80, 360, 100)),
        ('nose', (300, 100, 340, 120)),
        ('mouth', (300, 120, 340, 140)),
        # 身体区域
        ('head', (270, 50, 370, 150)),
        ('torso', (270, 150, 370, 250)),
        ('left_shoulder', (220, 180, 270, 220)),
        ('right_shoulder', (370, 180, 420, 220)),
        ('left_arm', (220, 150, 320, 250)),
        ('right_arm', (370, 150, 470, 250))
    ]

    # 设置 mock 对象的行为
    mock_camera_manager.read_frame.return_value = reference_frame
    mock_detector.detect_pose.return_value = True
    mock_detector.detect_face.return_value = True
    
    # 设置 mock_frame_processor 的行为
    mock_frame_processor.regions = []  # 初始化空列表
    
    def process_frame_effect(frame):
        """模拟帧处理效果"""
        # 确保 regions 属性已经被正确设置
        if not mock_frame_processor.regions:
            # 如果 regions 为空，调用 create_multi_regions 创建区域
            mock_frame_processor.regions = create_multi_regions(None, frame.shape)
        return frame  # 简单返回原始帧用于测试
        
    mock_frame_processor.process_frame = Mock(side_effect=process_frame_effect)
    
    with app.app_context():
        with patch('run.camera_manager', mock_camera_manager), \
             patch('run.pose', mock_pose), \
             patch('run.face_mesh', mock_face_mesh), \
             patch('run.pose_binding', mock_pose_binding), \
             patch('run.frame_processor', mock_frame_processor), \
             patch('run.detector', mock_detector), \
             patch('run.pose_deformer', mock_pose_deformer):
            
            print("\n=== 测试初始化 ===")
            print(f"参考帧尺寸: {reference_frame.shape}")
            print(f"当前帧尺寸: {current_frame.shape}")
            
            # 验证初始状态
            print("\n=== 初始状态验证 ===")
            print(f"Frame processor regions: {len(mock_frame_processor.regions)}")
            print(f"Reference frame: {'已设置' if mock_frame_processor.reference_frame is not None else '未设置'}")
            
            # 捕获参考帧
            response = capture_reference()
            response, status_code = response
            response_data = response.get_json()
            
            print("\n=== 捕获响应分析 ===")
            print(f"状态码: {status_code}")
            print(f"成功标志: {response_data.get('success')}")
            print(f"区域信息: {response_data.get('details', {}).get('regions_info', {})}")
            
            # 验证捕获成功
            assert status_code == 200, "捕获参考帧应该成功"
            assert response_data['success'] is True, "捕获参考帧应该返回成功"
            assert 'regions_info' in response_data['details'], "应该包含变形区域信息"
            
            # 验证参考数据保存
            assert mock_frame_processor.reference_frame is not None, "参考帧应该被保存"
            assert mock_frame_processor.reference_pose is not None, "参考姿态应该被保存"
            assert mock_frame_processor.regions is not None, "变形区域应该被保存"
            
            # 验证参考帧的内容
            assert np.array_equal(mock_frame_processor.reference_frame, reference_frame), \
                "保存的参考帧应该与原始帧相同"
            
            # 6. 处理当前帧
            deformed_frame = process_frame(current_frame)
            
            # 验证变形结果
            assert deformed_frame is not None, "应该返回变形后的帧"
            assert deformed_frame.shape == current_frame.shape, "变形后的帧尺寸应该保持不变"
            assert deformed_frame.dtype == current_frame.dtype, "变形后的帧类型应该保持不变"
            
            # 验证变形效果
            assert np.any(deformed_frame != current_frame), "变形后的帧应该与原始帧不同"
            
            # 计算帧差异的位置和程度
            diff = cv2.absdiff(deformed_frame, current_frame)
            movement = np.mean(diff)
            assert movement > 0, "应该检测到明显的变形效果"
            
            # 验证变形方向
            # 假设向右移动，计算左右两侧的差异
            left_diff = np.mean(diff[:, :320])
            right_diff = np.mean(diff[:, 320:])
            assert right_diff > left_diff, "变形应该向右移动"
            
            # 验证每个区域的变形效果
            print("\n=== 区域变形分析 ===")
            total_regions = len(regions_to_check)
            print(f"分析 {total_regions} 个主要区域")
            
            for name, (x1, y1, x2, y2) in regions_to_check:
                print(f"\n{name}区域 ({x1},{y1} - {x2},{y2}):")
                
                # 提取区域
                current_region = current_frame[y1:y2, x1:x2]
                deformed_region = deformed_frame[y1:y2, x1:x2]
                region_diff = cv2.absdiff(deformed_region, current_region)
                
                # 计算变形统计
                movement = np.mean(region_diff)
                left_diff = np.mean(region_diff[:, :region_diff.shape[1]//2])
                right_diff = np.mean(region_diff[:, region_diff.shape[1]//2:])
                max_diff = np.max(region_diff)
                
                print(f"  区域尺寸: {current_region.shape}")
                print(f"  整体变形程度: {movement:.2f}")
                print(f"  最大变形程度: {max_diff:.2f}")
                print(f"  左侧变形程度: {left_diff:.2f}")
                print(f"  右侧变形程度: {right_diff:.2f}")
                print(f"  变形方向: {'右' if right_diff > left_diff else '左'}")
                print(f"  方向差异: {abs(right_diff - left_diff):.2f}")
                
                # 保存区域分析图
                region_vis = np.hstack([
                    current_region,
                    deformed_region,
                    cv2.applyColorMap(region_diff.astype(np.uint8), cv2.COLORMAP_JET)
                ])
                save_debug_image(f"region_analysis_{name}", region_vis)
                
                # 验证变形效果
                assert movement > 0, f"{name}区域应该有明显的变形效果"
                assert right_diff > left_diff, f"{name}区域应该向右移动"
            
            # 验证整体变形效果
            total_movement = np.mean(diff_frame)
            max_movement = np.max(diff_frame)
            movement_std = np.std(diff_frame)
            
            print("\n=== 整体变形统计 ===")
            print(f"平均变形程度: {total_movement:.2f}")
            print(f"最大变形程度: {max_movement:.2f}")
            print(f"变形标准差: {movement_std:.2f}")
            print(f"非零变形像素比例: {np.count_nonzero(diff_frame) / diff_frame.size:.2%}")
            
            assert total_movement > 0, "应该有明显的整体变形效果"
            
            # 添加更多验证点
            print("\n=== 变形区域详细验证 ===")
            for region in mock_frame_processor.regions:
                print(f"\n区域: {region.name}")
                print(f"类型: {region.type}")
                print(f"中心点: {region.center}")
                print(f"遮罩尺寸: {region.mask.shape}")
                print(f"绑定点数量: {len(region.binding_points)}")
                
                # 验证绑定点的有效性
                for i, point in enumerate(region.binding_points):
                    assert hasattr(point, 'landmark_index'), f"绑定点 {i} 缺少 landmark_index"
                    assert hasattr(point, 'local_coords'), f"绑定点 {i} 缺少 local_coords"
                    print(f"  点 {i}: 索引={point.landmark_index}, 坐标={point.local_coords}")
            
            # ... (后面的代码保持不变) 
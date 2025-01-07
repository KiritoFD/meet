import numpy as np
from typing import Dict, List, Optional, Tuple
import cv2
import logging
from dataclasses import dataclass
import time
import json
import os
import uuid

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    model_path: str
    texture_path: Optional[str] = None
    scale: float = 1.0
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    enable_physics: bool = False
    enable_animation: bool = True
    lod_levels: int = 3

@dataclass
class Model:
    id: str
    config: ModelConfig
    vertices: np.ndarray
    faces: np.ndarray
    texture: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    uv_coords: Optional[np.ndarray] = None
    skeleton: Optional[Dict] = None
    animations: Optional[Dict] = None
    transform_matrix: np.ndarray = None
    
    def __post_init__(self):
        if self.transform_matrix is None:
            self.transform_matrix = np.eye(4)

class ModelManager:
    def __init__(self):
        """初始化模型管理器"""
        self.models: Dict[str, Model] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.cache_dir = "model_cache"
        self.max_cache_size = 1024 * 1024 * 1024  # 1GB
        
        # 创建缓存目录
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.recording_data = {}  # 存储录制数据
        self.calibration_poses = {
            'T': None,  # T-pose数据
            'A': None,  # A-pose数据
            'N': None   # 自然站姿数据
        }
        
    def load_model(self, model_id: str, config: ModelConfig) -> Model:
        """加载3D模型"""
        try:
            # 1. 检查缓存
            cached_model = self._load_from_cache(model_id)
            if cached_model:
                logger.info(f"从缓存加载模型: {model_id}")
                return cached_model
                
            # 2. 加载模型文件
            vertices, faces = self._load_model_file(config.model_path)
            
            # 3. 加载纹理
            texture = None
            if config.texture_path:
                texture = self._load_texture(config.texture_path)
                
            # 4. 计算法线
            normals = self._calculate_normals(vertices, faces)
            
            # 5. 创建模型对象
            model = Model(
                id=model_id,
                config=config,
                vertices=vertices,
                faces=faces,
                texture=texture,
                normals=normals
            )
            
            # 6. 应用变换
            self._apply_transform(model)
            
            # 7. 缓存模型
            self._save_to_cache(model)
            
            # 8. 保存到内存
            self.models[model_id] = model
            self.model_configs[model_id] = config
            
            logger.info(f"加载模型成功: {model_id}")
            return model
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
        
    def unload_model(self, model_id: str):
        """卸载模型"""
        try:
            if model_id in self.models:
                del self.models[model_id]
                del self.model_configs[model_id]
                logger.info(f"卸载模型成功: {model_id}")
                
        except Exception as e:
            logger.error(f"卸载模型失败: {e}")
        
    def update_model(self, model_id: str, transform: Dict):
        """更新模型变换"""
        try:
            if model_id not in self.models:
                raise ValueError(f"模型不存在: {model_id}")
                
            model = self.models[model_id]
            
            # 更新变换矩阵
            if "position" in transform:
                model.config.position = transform["position"]
            if "rotation" in transform:
                model.config.rotation = transform["rotation"]
            if "scale" in transform:
                model.config.scale = transform["scale"]
                
            # 重新应用变换
            self._apply_transform(model)
            
            logger.info(f"更新模型变换成功: {model_id}")
            
        except Exception as e:
            logger.error(f"更新模型变换失败: {e}")
        
    def apply_animation(self, model_id: str, animation_data: Dict):
        """应用动画数据"""
        try:
            if model_id not in self.models:
                raise ValueError(f"模型不存在: {model_id}")
                
            model = self.models[model_id]
            if not model.config.enable_animation:
                return
                
            # TODO: 实现以下功能
            # 1. 解析动画数据
            # 2. 更新骨骼状态
            # 3. 计算顶点变形
            # 4. 更新模型缓冲
            
            logger.info(f"应用动画成功: {model_id}")
            
        except Exception as e:
            logger.error(f"应用动画失败: {e}")
        
    def _load_model_file(self, model_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载模型文件"""
        try:
            # TODO: 实现以下功能
            # 1. 支持多种格式(.obj, .fbx, .gltf)
            # 2. 读取顶点数据
            # 3. 读取面片数据
            # 4. 读取UV坐标
            # 5. 读取骨骼数据
            
            # 临时返回空数据
            vertices = np.array([])
            faces = np.array([])
            return vertices, faces
            
        except Exception as e:
            logger.error(f"加载模型文件失败: {e}")
            raise
        
    def _load_texture(self, texture_path: str) -> Optional[np.ndarray]:
        """加载纹理"""
        try:
            if not os.path.exists(texture_path):
                raise FileNotFoundError(f"纹理文件不存在: {texture_path}")
                
            texture = cv2.imread(texture_path)
            if texture is None:
                raise ValueError(f"无法读取纹理文件: {texture_path}")
                
            return cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            logger.error(f"加载纹理失败: {e}")
            return None
        
    def _calculate_normals(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """计算法线"""
        try:
            # TODO: 实现以下功能
            # 1. 计算面片法线
            # 2. 计算顶点法线
            # 3. 法线平滑
            return np.zeros_like(vertices)
            
        except Exception as e:
            logger.error(f"计算法线失败: {e}")
            return np.array([])
        
    def _apply_transform(self, model: Model):
        """应用变换"""
        try:
            # 1. 创建变换矩阵
            scale_matrix = np.eye(4)
            scale_matrix[0:3, 0:3] *= model.config.scale
            
            rotation_x = np.eye(4)
            rotation_y = np.eye(4)
            rotation_z = np.eye(4)
            
            # TODO: 填充旋转矩阵
            
            translation_matrix = np.eye(4)
            translation_matrix[0:3, 3] = model.config.position
            
            # 2. 组合变换
            model.transform_matrix = translation_matrix @ rotation_z @ rotation_y @ rotation_x @ scale_matrix
            
        except Exception as e:
            logger.error(f"应用变换失败: {e}")
        
    def _load_from_cache(self, model_id: str) -> Optional[Model]:
        """从缓存加载模型"""
        try:
            cache_path = os.path.join(self.cache_dir, f"{model_id}.cache")
            if not os.path.exists(cache_path):
                return None
                
            # TODO: 实现缓存加载
            return None
            
        except Exception as e:
            logger.error(f"从缓存加载失败: {e}")
            return None
        
    def _save_to_cache(self, model: Model):
        """保存模型到缓存"""
        try:
            # 检查缓存大小
            self._cleanup_cache()
            
            cache_path = os.path.join(self.cache_dir, f"{model.id}.cache")
            
            # TODO: 实现缓存保存
            
        except Exception as e:
            logger.error(f"保存到缓存失败: {e}")
        
    def _cleanup_cache(self):
        """清理缓存"""
        try:
            # 获取所有缓存文件
            cache_files = []
            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".cache"):
                    file_path = os.path.join(self.cache_dir, filename)
                    cache_files.append((file_path, os.path.getmtime(file_path)))
                    
            # 按修改时间排序
            cache_files.sort(key=lambda x: x[1])
            
            # 计算总大小
            total_size = sum(os.path.getsize(f[0]) for f in cache_files)
            
            # 删除旧文件直到低于最大缓存大小
            while total_size > self.max_cache_size and cache_files:
                file_path, _ = cache_files.pop(0)
                total_size -= os.path.getsize(file_path)
                os.remove(file_path)
                logger.info(f"清理缓存文件: {file_path}")
                
        except Exception as e:
            logger.error(f"清理缓存失败: {e}")
        
    def __del__(self):
        """清理资源"""
        # 保存所有模型到缓存
        for model in self.models.values():
            try:
                self._save_to_cache(model)
            except:
                pass 
        
    def start_recording(self, user_id: str) -> None:
        """开始录制用户动作"""
        self.recording_data[user_id] = {
            'frames': [],
            'start_time': time.time(),
            'calibrated': False
        }
        logger.info(f"开始录制用户动作: {user_id}")
        
    def stop_recording(self, user_id: str) -> Dict:
        """停止录制并返回录制数据"""
        try:
            if user_id not in self.recording_data:
                raise ValueError(f"未找到用户录制数据: {user_id}")
                
            recording = self.recording_data[user_id]
            duration = time.time() - recording['start_time']
            
            # 处理录制数据
            processed_data = self._process_recording(recording['frames'])
            
            # 清理录制数据
            del self.recording_data[user_id]
            
            return {
                'status': 'success',
                'frames': processed_data,
                'duration': duration
            }
            
        except Exception as e:
            logger.error(f"停止录制失败: {e}")
            raise
            
    def add_calibration_pose(self, user_id: str, pose_type: str, pose_data: Dict) -> None:
        """添加校准姿势数据"""
        try:
            if pose_type not in ['T', 'A', 'N']:
                raise ValueError(f"不支持的姿势类型: {pose_type}")
                
            if user_id not in self.recording_data:
                self.start_recording(user_id)
                
            # 保存校准数据
            self.calibration_poses[pose_type] = pose_data
            
            # 检查是否完成所有校准
            if all(pose is not None for pose in self.calibration_poses.values()):
                self.recording_data[user_id]['calibrated'] = True
                logger.info(f"用户校准完成: {user_id}")
                
        except Exception as e:
            logger.error(f"添加校准姿势失败: {e}")
            raise
            
    def add_frame(self, user_id: str, frame_data: Dict) -> None:
        """添加录制帧"""
        try:
            if user_id not in self.recording_data:
                raise ValueError(f"未开始录制: {user_id}")
                
            # 添加时间戳
            frame_data['timestamp'] = time.time()
            
            # 保存帧数据
            self.recording_data[user_id]['frames'].append(frame_data)
            
        except Exception as e:
            logger.error(f"添加录制帧失败: {e}")
            raise
            
    def _process_recording(self, frames: List[Dict]) -> List[Dict]:
        """处理录制数据"""
        try:
            # 1. 数据平滑
            smoothed_frames = self._smooth_frames(frames)
            
            # 2. 关键帧提取
            key_frames = self._extract_key_frames(smoothed_frames)
            
            # 3. 动作规范化
            normalized_frames = self._normalize_frames(key_frames)
            
            return normalized_frames
            
        except Exception as e:
            logger.error(f"处理录制数据失败: {e}")
            raise
            
    def _smooth_frames(self, frames: List[Dict]) -> List[Dict]:
        """平滑帧数据"""
        try:
            window_size = 5
            smoothed = []
            
            for i in range(len(frames)):
                # 获取窗口范围内的帧
                start = max(0, i - window_size // 2)
                end = min(len(frames), i + window_size // 2 + 1)
                window = frames[start:end]
                
                # 计算平均值
                smoothed_frame = self._average_frames(window)
                smoothed.append(smoothed_frame)
                
            return smoothed
            
        except Exception as e:
            logger.error(f"平滑帧数据失败: {e}")
            raise 

    def _convert_recording_to_model(self, recording_data: Dict) -> Dict:
        """将录制数据转换为3D模型数据"""
        try:
            # 1. 从校准数据构建基础骨骼
            skeleton = self._build_skeleton_from_calibration()
            
            # 2. 从录制帧生成顶点和面片
            vertices, faces = self._generate_mesh_from_frames(recording_data['frames'])
            
            # 3. 生成动画数据
            animations = self._generate_animations(recording_data['frames'], skeleton)
            
            return {
                'model_id': str(uuid.uuid4()),
                'vertices': vertices,
                'faces': faces,
                'skeleton': skeleton,
                'animations': animations
            }
            
        except Exception as e:
            logger.error(f"转换模型数据失败: {e}")
            raise

    def _build_skeleton_from_calibration(self) -> Dict:
        """从校准数据构建骨骼结构"""
        try:
            # 使用T-pose作为基准姿势
            t_pose = self.calibration_poses['T']
            
            # 定义关节层级
            joint_hierarchy = {
                'hips': ['spine', 'left_hip', 'right_hip'],
                'spine': ['chest', 'neck'],
                'neck': ['head', 'left_shoulder', 'right_shoulder'],
                # ... 更多关节定义
            }
            
            # 计算关节位置和方向
            joints = self._calculate_joints(t_pose['pose'])
            
            return {
                'joints': joints,
                'hierarchy': joint_hierarchy
            }
            
        except Exception as e:
            logger.error(f"构建骨骼失败: {e}")
            raise

    def _generate_mesh_from_frames(self, frames: List[Dict]) -> Tuple[List[float], List[int]]:
        """从录制帧生成网格数据"""
        try:
            # 1. 合并所有帧的点云数据
            point_cloud = self._merge_frame_points(frames)
            
            # 2. 点云降采样和平滑
            filtered_points = self._filter_point_cloud(point_cloud)
            
            # 3. 生成三角面片
            vertices, faces = self._triangulate_points(filtered_points)
            
            # 4. 优化网格
            vertices, faces = self._optimize_mesh(vertices, faces)
            
            return vertices, faces
            
        except Exception as e:
            logger.error(f"生成网格失败: {e}")
            raise 
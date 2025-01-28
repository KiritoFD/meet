import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
import json
import open3d as o3d
from typing import List, Optional
import time

@dataclass
class PersonalModel:
    """个人3D面部模型"""
    mesh: o3d.geometry.TriangleMesh
    texture: np.ndarray
    landmarks_mean: np.ndarray
    landmarks_std: np.ndarray
    displacement: np.ndarray
    normal_map: np.ndarray

@dataclass
class FacialFeatures:
    """面部特征数据"""
    landmarks: np.ndarray
    timestamp: float
    blendshapes: Optional[dict] = None  # 表情系数

class FaceModelSystem:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # 加载预定义的面部拓扑结构
        self.FACE_TRIANGLES = mp.solutions.face_mesh_connections.FACEMESH_TESSELATION
        
        # 表情系数定义
        self.BLENDSHAPES = {
            'smile': [61, 291, 39, 181],  # 微笑相关的特征点
            'eye_blink': [386, 374, 159, 145],  # 眨眼相关的特征点
            'mouth_open': [13, 14, 78, 308],  # 张嘴相关的特征点
        }
        
        self.personal_model = None
    
    def create_personal_model(self, features_list: List[FacialFeatures]) -> PersonalModel:
        """从多帧特征创建个人模型"""
        print("正在生成个人模型...")
        
        # 1. 计算平均特征点位置
        all_landmarks = np.stack([f.landmarks for f in features_list])
        landmarks_mean = np.mean(all_landmarks, axis=0)
        landmarks_std = np.std(all_landmarks, axis=0)
        
        # 2. 创建3D网格
        vertices = landmarks_mean
        triangles = np.array([[conn.p1, conn.p2] for conn in self.FACE_TRIANGLES])
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        
        # 3. 计算法线和UV映射
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        
        # 4. 生成基础纹理
        texture = self._generate_base_texture(features_list[0])
        
        print("个人模型生成完成！")
        return PersonalModel(mesh, texture, landmarks_mean, landmarks_std, np.zeros((512, 512, 3)), np.zeros((512, 512, 3)))
    
    def _generate_base_texture(self, features: FacialFeatures) -> np.ndarray:
        """生成基础纹理图"""
        # 简单的纹理生成示例
        texture = np.zeros((512, 512, 3), dtype=np.uint8)
        # TODO: 实现更复杂的纹理生成
        return texture
    
    def extract_features(self, frame) -> FacialFeatures:
        """提取面部特征"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        landmarks = np.array([[lm.x, lm.y, lm.z] 
                            for lm in results.multi_face_landmarks[0].landmark])
        
        # 计算表情系数
        blendshapes = self._calculate_blendshapes(landmarks)
        
        return FacialFeatures(
            landmarks=landmarks,
            timestamp=time.time(),
            blendshapes=blendshapes
        )
    
    def _calculate_blendshapes(self, landmarks: np.ndarray) -> dict:
        """计算表情系数"""
        blendshapes = {}
        
        for name, points in self.BLENDSHAPES.items():
            # 计算特定点之间的距离变化来估计表情系数
            distances = []
            for i in range(0, len(points), 2):
                p1 = landmarks[points[i]]
                p2 = landmarks[points[i+1]]
                distances.append(np.linalg.norm(p1 - p2))
            blendshapes[name] = np.mean(distances)
        
        return blendshapes
    
    def render_frame(self, features: FacialFeatures) -> np.ndarray:
        """渲染一帧"""
        if self.personal_model is None:
            return None
            
        # 1. 更新模型顶点位置
        updated_vertices = self._update_vertices(features)
        self.personal_model.mesh.vertices = o3d.utility.Vector3dVector(updated_vertices)
        
        # 2. 应用表情变形
        if features.blendshapes:
            self._apply_blendshapes(features.blendshapes)
        
        # 3. 渲染结果
        return self._render_mesh()
    
    def _update_vertices(self, features: FacialFeatures) -> np.ndarray:
        """根据特征点更新顶点位置"""
        # 计算特征点的偏移
        offset = features.landmarks - self.personal_model.landmarks_mean
        normalized_offset = offset / (self.personal_model.landmarks_std + 1e-6)
        
        # 应用偏移到模型顶点
        vertices = np.asarray(self.personal_model.mesh.vertices)
        updated_vertices = vertices + normalized_offset
        return updated_vertices
    
    def _apply_blendshapes(self, blendshapes: dict):
        """应用表情变形"""
        # TODO: 实现基于blendshapes的表情变形
        pass
    
    def _render_mesh(self) -> np.ndarray:
        """渲染3D网格"""
        # 使用Open3D渲染
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(self.personal_model.mesh)
        
        # 设置相机参数
        ctr = vis.get_view_control()
        ctr.set_front([0, 0, -1])
        ctr.set_up([0, -1, 0])
        
        # 渲染
        vis.poll_events()
        vis.update_renderer()
        image = vis.capture_screen_float_buffer()
        vis.destroy_window()
        
        return np.asarray(image)

    def create_detailed_mesh(self, features_list: List[FacialFeatures]) -> PersonalModel:
        """创建高精度人体网格"""
        # 1. 生成基础网格
        base_mesh = self._create_base_mesh(features_list)
        
        # 2. 添加细节层
        displacement_map = self._calculate_displacement(base_mesh, features_list)
        normal_map = self._calculate_normal_map(displacement_map)
        
        # 3. 多分辨率处理
        subdivided_mesh = base_mesh.subdivide_loop(number_of_cuts=3)
        subdivided_mesh.vertices += displacement_map * 0.1  # 应用位移贴图
        
        # 4. 生成UV映射
        self._generate_uv_map(subdivided_mesh)
        
        return PersonalModel(
            mesh=subdivided_mesh,
            texture=self._generate_pbr_texture(normal_map),
            displacement=displacement_map,
            normal_map=normal_map
        )

    def _reconstruct_surface(self, point_cloud):
        """泊松表面重建"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        
        # 执行泊松重建
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9, linear_fit=True
        )
        
        # 移除低密度区域
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        return mesh

def main():
    system = FaceModelSystem()
    
    # 1. 捕获训练数据
    print("正在捕获训练数据...")
    cap = cv2.VideoCapture(0)
    training_features = []
    
    for _ in range(30):  # 捕获30帧
        ret, frame = cap.read()
        if not ret:
            break
            
        features = system.extract_features(frame)
        if features is not None:
            training_features.append(features)
            cv2.imshow('Capturing', frame)
            cv2.waitKey(1)
    
    # 2. 创建个人模型
    system.personal_model = system.create_personal_model(training_features)
    
    # 3. 实时追踪和渲染
    print("\n开始实时追踪和渲染...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        features = system.extract_features(frame)
        if features is not None:
            # 渲染结果
            rendered = system.render_frame(features)
            if rendered is not None:
                # 显示原始图像和渲染结果
                cv2.imshow('Original', frame)
                cv2.imshow('Rendered', rendered)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
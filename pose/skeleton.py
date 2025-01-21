from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from scipy.spatial.transform import Rotation

@dataclass
class Joint:
    """关节类"""
    name: str
    parent_id: int  # -1 表示根节点
    initial_position: np.ndarray  # 初始局部位置
    initial_rotation: np.ndarray  # 初始局部旋转
    children: List[int]  # 子关节索引列表
    bind_matrix: np.ndarray  # 绑定姿态矩阵

class Skeleton:
    """骨骼系统"""
    def __init__(self):
        self.joints: List[Joint] = []
        self.joint_matrices: np.ndarray = None  # 关节变换矩阵
        self.inverse_bind_matrices: np.ndarray = None  # 逆绑定矩阵
        
    def add_joint(self, joint: Joint):
        """添加关节"""
        self.joints.append(joint)
        
    def update_matrices(self):
        """更新所有关节矩阵"""
        if not self.joints:
            return
            
        self.joint_matrices = np.zeros((len(self.joints), 4, 4))
        self.inverse_bind_matrices = np.zeros((len(self.joints), 4, 4))
        
        for i, joint in enumerate(self.joints):
            self.inverse_bind_matrices[i] = np.linalg.inv(joint.bind_matrix)
            
        self._update_joint_hierarchy(0)
        
    def _update_joint_hierarchy(self, joint_idx: int, parent_matrix: np.ndarray = None):
        """递归更新关节层级"""
        joint = self.joints[joint_idx]
        local_matrix = self._compute_local_matrix(joint)
        
        if parent_matrix is None:
            self.joint_matrices[joint_idx] = local_matrix
        else:
            self.joint_matrices[joint_idx] = parent_matrix @ local_matrix
            
        for child_idx in joint.children:
            self._update_joint_hierarchy(child_idx, self.joint_matrices[joint_idx])
            
    def _compute_local_matrix(self, joint: Joint) -> np.ndarray:
        """计算局部变换矩阵"""
        rotation_matrix = Rotation.from_euler('xyz', joint.initial_rotation).as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = joint.initial_position
        return transform 
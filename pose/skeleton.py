from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from scipy.spatial.transform import Rotation

@dataclass
class Joint:
    """关节类"""
    name: str
    parent_id: int  # -1 表示根节点
    position: np.ndarray  # 关节本地坐标
    rotation: np.ndarray  # 欧拉角 (弧度)
    children: List[int]  # 子关节索引列表
    bind_matrix: np.ndarray = np.eye(4)  # 绑定矩阵

class Skeleton:
    """骨骼系统"""
    def __init__(self):
        self.joints: List[Joint] = []
        self.joint_matrices: np.ndarray = None  # 关节变换矩阵
        
    def add_joint(self, joint: Joint):
        """添加关节"""
        self.joints.append(joint)
        
    def update_matrices(self):
        """更新所有关节矩阵"""
        # 递归计算关节的全局变换矩阵
        def _update_recursive(joint_idx, parent_matrix):
            joint = self.joints[joint_idx]
            local_matrix = self._compute_local_matrix(joint)
            global_matrix = parent_matrix @ local_matrix
            self.joint_matrices[joint_idx] = global_matrix
            for child_idx in joint.children:
                _update_recursive(child_idx, global_matrix)
        
        if self.joints:
            self.joint_matrices = np.zeros((len(self.joints), 4, 4))
            _update_recursive(0, np.eye(4))
        
    def _compute_local_matrix(self, joint: Joint) -> np.ndarray:
        """计算局部变换矩阵"""
        rotation_matrix = Rotation.from_euler('xyz', joint.rotation).as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = joint.position
        return transform 
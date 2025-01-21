import numpy as np
from math import radians

class Camera:
    def __init__(self, position=(0, 0, 3), target=(0, 0, 0), up=(0, 1, 0)):
        self.position = np.array(position, dtype=np.float32)
        self.target = np.array(target, dtype=np.float32)
        self.up = np.array(up, dtype=np.float32)
        
    def get_view_matrix(self):
        """获取视图矩阵"""
        z = self.position - self.target
        z = z / np.linalg.norm(z)
        
        x = np.cross(self.up, z)
        x = x / np.linalg.norm(x)
        
        y = np.cross(z, x)
        
        view_matrix = np.identity(4, dtype=np.float32)
        view_matrix[:3, 0] = x
        view_matrix[:3, 1] = y
        view_matrix[:3, 2] = z
        view_matrix[3, :3] = -np.dot([x, y, z], self.position)
        
        return view_matrix 
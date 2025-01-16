import numpy as np

class PoseTransformer:
    def __init__(self, max_cache_size=30):
        self.last_matrix = None
        self.last_result = None
        self.pose_cache = []
        self.max_cache_size = max_cache_size
        
    def update_cache(self, pose_data):
        """更新姿态缓存"""
        self.pose_cache.append(pose_data)
        if len(self.pose_cache) > self.max_cache_size:
            self.pose_cache.pop(0)
            
    def get_smoothed_pose(self):
        """获取平滑后的姿态数据"""
        if not self.pose_cache:
            return None
            
        # 使用最近的几帧计算平均姿态
        recent_poses = self.pose_cache[-5:]
        if not recent_poses:
            return self.pose_cache[-1]
            
        avg_pose = []
        for i in range(len(recent_poses[0])):
            point_data = np.mean([pose[i] for pose in recent_poses], axis=0)
            avg_pose.append(point_data.tolist())
            
        return avg_pose 
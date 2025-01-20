class PoseBinder:
    def __init__(self):
        # 添加区域权重配置
        self._region_weights = {
            'torso': 1.0,
            'left_arm': 0.8,
            'right_arm': 0.8,
            'left_leg': 0.8,
            'right_leg': 0.8
        }
        
        # 添加区域混合参数
        self._overlap_radius = 20  # 区域重叠半径
        self._min_points = 3      # 每个区域最少绑定点数

    def generate_binding_points(self, frame_shape, regions):
        # 添加密度控制
        min_spacing = 10  # 最小采样间距
        max_spacing = 30  # 最大采样间距
        
        # 根据区域大小自适应采样密度
        for region in regions.values():
            area = cv2.contourArea(region.contour)
            spacing = np.clip(np.sqrt(area) / 20, 
                            min_spacing, max_spacing)
            points = self._grid_sample(region, spacing) 
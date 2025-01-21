import math
import random
import pygame
import numpy as np
from pygame import gfxdraw

class FaceVisualizer:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("高精细卡通渲染")
        
        # 颜色系统
        self.colors = {
            'skin': {
                'base': (255, 226, 205),
                'shadow': [(245, 206, 185), (235, 186, 165), (225, 166, 145)],
                'highlight': [(255, 236, 215), (255, 246, 225)],
                'outline': (180, 140, 120, 150),
                'blush': [(255, 180, 180, 100), (255, 150, 150, 80)],
                'texture': (245, 216, 195, 5)
            },
            'eyes': {
                'white': (255, 255, 255),
                'iris': {
                    'blue': [(92, 160, 216), (72, 140, 196), (52, 120, 176)],
                    'brown': [(139, 69, 19), (119, 49, 0), (99, 29, 0)],
                    'green': [(34, 139, 34), (14, 119, 14), (0, 99, 0)]
                },
                'pupil': (20, 20, 20),
                'highlight': [(255, 255, 255, 220), (255, 255, 255, 180)],
                'shadow': [(180, 180, 180), (160, 160, 160)],
                'outline': (60, 60, 60),
                'lashes': (40, 40, 40)
            },
            'hair': {
                'base': [(40, 30, 20), (35, 25, 15)],
                'highlight': [(90, 70, 50), (70, 50, 35), (50, 35, 25)],
                'shadow': [(30, 20, 15), (20, 15, 10)],
                'outline': (20, 15, 10),
                'strand': [(45, 35, 25, 150), (35, 25, 15, 100)]
            },
            'lips': {
                'base': [(255, 150, 150), (235, 130, 130)],
                'highlight': [(255, 180, 180), (255, 160, 160)],
                'shadow': [(215, 110, 110), (195, 90, 90)],
                'outline': (180, 100, 100, 150)
            }
        }

    def draw_face(self, values):
        """主绘制函数"""
        # 清空屏幕
        self.screen.fill((240, 240, 240))
        
        # 创建图层
        layers = {
            'base': pygame.Surface((self.width, self.height), pygame.SRCALPHA),
            'features': pygame.Surface((self.width, self.height), pygame.SRCALPHA),
            'hair': pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        }
        
        # 1. 绘制基础脸型
        face_shape = self.generate_advanced_face_shape(values)
        self.draw_advanced_face(layers['base'], face_shape, values)
        
        # 2. 绘制五官（相对位置）
        center_x = self.width // 2
        center_y = self.height // 2
        
        # 眼睛位置（在脸部上三分之一处）
        eye_y = center_y - 30
        eye_spacing = 60
        self.draw_advanced_eyes(layers['features'], 
                              [(center_x - eye_spacing, eye_y),
                               (center_x + eye_spacing, eye_y)],
                              values)
        
        # 眉毛位置（眼睛上方）
        brow_y = eye_y - 30
        self.draw_advanced_eyebrows(layers['features'],
                                  [(center_x - eye_spacing, brow_y),
                                   (center_x + eye_spacing, brow_y)],
                                  values)
        
        # 鼻子位置（脸部中心略上）
        nose_y = center_y + 10
        self.draw_advanced_nose(layers['features'],
                              (center_x, nose_y),
                              values)
        
        # 嘴巴位置（下三分之一处）
        mouth_y = center_y + 50
        self.draw_advanced_mouth(layers['features'],
                               (center_x, mouth_y),
                               values)
        
        # 3. 合并所有图层
        for layer in layers.values():
            self.screen.blit(layer, (0, 0))
        
        pygame.display.flip()

    def generate_advanced_face_shape(self, values):
        """生成标准人脸轮廓"""
        center_x = self.width // 2
        center_y = self.height // 2
        
        # 基础参数（标准人脸比例）
        face_width = 180  # 脸宽
        face_height = 240  # 脸高
        jaw_width = 160   # 下巴宽度
        
        # 定义关键点
        points = []
        
        # 1. 额头曲线（3点）
        points.extend([
            (center_x - face_width//2, center_y - face_height//3),  # 左额角
            (center_x, center_y - face_height//2),                  # 额头中点
            (center_x + face_width//2, center_y - face_height//3),  # 右额角
        ])
        
        # 2. 两侧脸颊（4点）
        points.extend([
            (center_x + face_width//2, center_y),                   # 右脸中部
            (center_x + jaw_width//2, center_y + face_height//4),   # 右下颌
            (center_x - jaw_width//2, center_y + face_height//4),   # 左下颌
            (center_x - face_width//2, center_y),                   # 左脸中部
        ])
        
        # 3. 下巴（3点）
        points.extend([
            (center_x - jaw_width//3, center_y + face_height//3),   # 左下巴
            (center_x, center_y + face_height//2.5),                # 下巴尖
            (center_x + jaw_width//3, center_y + face_height//3),   # 右下巴
        ])
        
        # 使用贝塞尔曲线平滑连接所有点
        smooth_points = []
        segments = len(points) - 2
        steps_per_segment = 10
        
        for i in range(segments):
            p1 = points[i]
            p2 = points[i + 1]
            p3 = points[i + 2]
            
            for step in range(steps_per_segment):
                t = step / steps_per_segment
                # 二次贝塞尔曲线
                x = (1-t)**2 * p1[0] + 2*(1-t)*t * p2[0] + t**2 * p3[0]
                y = (1-t)**2 * p1[1] + 2*(1-t)*t * p2[1] + t**2 * p3[1]
                smooth_points.append((int(x), int(y)))
        
        return smooth_points

    def draw_advanced_face(self, surface, face_shape, values):
        """绘制精细的面部"""
        # 1. 基础肤色
        self.draw_smooth_polygon(surface, self.colors['skin']['base'], face_shape)
        
        # 2. 添加立体感阴影
        shadow_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for i, shadow_color in enumerate(self.colors['skin']['shadow']):
            offset = i * 5
            shadow_points = [(x+offset, y+offset) for x, y in face_shape]
            self.draw_smooth_polygon(shadow_surface, (*shadow_color, 50), shadow_points)
        surface.blit(shadow_surface, (0, 0))
        
        # 3. 添加高光
        highlight_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for i, highlight_color in enumerate(self.colors['skin']['highlight']):
            offset = i * -3
            highlight_points = [(x+offset, y+offset) for x, y in face_shape]
            self.draw_smooth_polygon(highlight_surface, (*highlight_color, 40), highlight_points)
        surface.blit(highlight_surface, (0, 0))
        
        # 4. 添加肤色纹理
        self.add_skin_texture(surface, face_shape)
        
        # 5. 添加腮红
        self.add_blush(surface, values)

    def add_skin_texture(self, surface, face_shape):
        """添加皮肤纹理"""
        texture_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # 创建噪点纹理
        for _ in range(2000):
            x = random.randint(0, self.width-1)
            y = random.randint(0, self.height-1)
            
            # 检查点是否在面部区域内
            if self.point_in_polygon((x, y), face_shape):
                alpha = random.randint(2, 5)
                color = (*self.colors['skin']['texture'][:3], alpha)
                
                # 绘制细小的纹理点
                size = random.uniform(0.5, 1.5)
                self.draw_smooth_circle(texture_surface, color, (x, y), size)
        
        surface.blit(texture_surface, (0, 0))

    def draw_back_hair(self, surface, values):
        """绘制后层头发"""
        center_x = self.width // 2
        center_y = self.height // 2
        
        # 创建基础头发形状
        hair_points = []
        steps = 40
        
        for i in range(steps):
            angle = 2 * math.pi * i / steps
            radius = 220  # 基础半径
            
            # 添加自然变化
            radius += math.sin(angle * 3) * 20
            radius += random.uniform(-10, 10)
            
            x = center_x + math.cos(angle) * radius
            y = center_y + math.sin(angle) * radius
            
            # 调整头发长度
            if math.pi/2 < angle < 3*math.pi/2:
                y += 100  # 加长后部头发
            
            hair_points.append((int(x), int(y)))
        
        # 绘制多层头发
        for i, base_color in enumerate(self.colors['hair']['base']):
            offset = i * 5
            points = [(x+random.uniform(-offset, offset), 
                      y+random.uniform(-offset, offset)) 
                     for x, y in hair_points]
            
            self.draw_smooth_polygon(surface, base_color, points)
        
        # 添加头发细节
        self.add_hair_details(surface, hair_points)

    def add_hair_details(self, surface, base_points):
        """添加头发细节"""
        detail_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # 添加发丝
        for _ in range(200):
            # 随机选择基准点
            idx = random.randint(0, len(base_points)-1)
            start_x, start_y = base_points[idx]
            
            # 发丝参数
            length = random.uniform(20, 40)
            angle = math.atan2(start_y - self.height//2, 
                             start_x - self.width//2)
            angle += random.uniform(-0.5, 0.5)
            
            # 计算发丝终点
            end_x = start_x + math.cos(angle) * length
            end_y = start_y + math.sin(angle) * length
            
            # 绘制发丝
            for color in self.colors['hair']['strand']:
                offset = random.uniform(-2, 2)
                points = [
                    (start_x + offset, start_y + offset),
                    (end_x + offset, end_y + offset)
                ]
                pygame.draw.aaline(detail_surface, color, *points)
        
        surface.blit(detail_surface, (0, 0))

    def draw_front_hair(self, surface, values):
        # Implementation of draw_front_hair method
        pass

    def draw_advanced_eyes(self, surface, eye_positions, values):
        """绘制精细的眼睛"""
        eye_open = values.get('eye_open', 1.0)
        eye_squint = values.get('eye_squint', 0.0)
        
        for pos in eye_positions:
            eye_surface = pygame.Surface((60, 40), pygame.SRCALPHA)
            center_x, center_y = 30, 20
            
            # 眼睛大小和形状
            width = 40
            height = 25 * eye_open
            
            # 1. 眼白
            eye_rect = pygame.Rect(
                center_x - width//2,
                center_y - height//2,
                width,
                height
            )
            self.draw_smooth_ellipse(eye_surface, self.colors['eyes']['white'], eye_rect)
            
            # 2. 虹膜
            iris_color = self.colors['eyes']['iris']['brown']  # 可以改变眼睛颜色
            iris_size = 18
            iris_rect = pygame.Rect(
                center_x - iris_size//2,
                center_y - iris_size//2,
                iris_size,
                iris_size
            )
            for i, color in enumerate(iris_color):
                size = iris_size - i*2
                self.draw_smooth_circle(eye_surface, color, 
                                     (center_x, center_y), size//2)
            
            # 3. 瞳孔
            pupil_size = 8
            self.draw_smooth_circle(eye_surface, self.colors['eyes']['pupil'],
                                  (center_x, center_y), pupil_size//2)
            
            # 4. 眼睑
            eyelid_points = []
            steps = 20
            for i in range(steps):
                t = i / (steps-1)
                x = center_x - width//2 + width * t
                
                # 上眼睑
                upper_y = center_y - height//2 + math.sin(t * math.pi) * (10 + eye_squint * 10)
                eyelid_points.append((x, upper_y))
                
                # 下眼睑
                if i < steps-1:  # 避免重复最后一个点
                    lower_y = center_y + height//2 - math.sin(t * math.pi) * 5
                    eyelid_points.insert(0, (x, lower_y))
            
            # 绘制眼睑
            pygame.draw.polygon(eye_surface, self.colors['skin']['base'], eyelid_points)
            
            # 5. 睫毛
            if eye_open > 0.5:
                for i in range(5):
                    t = (i + 1) / 6
                    base_x = center_x - width//2 + width * t
                    base_y = center_y - height//2
                    
                    length = 8 * (1 - abs(t - 0.5) * 2)  # 中间睫毛最长
                    angle = -math.pi/3  # 向上倾斜
                    
                    end_x = base_x + math.cos(angle) * length
                    end_y = base_y + math.sin(angle) * length
                    
                    pygame.draw.line(eye_surface, self.colors['eyes']['lashes'],
                                   (base_x, base_y), (end_x, end_y), 2)
            
            # 6. 眼角
            for side in [-1, 1]:
                corner_x = center_x + side * width//2
                corner_y = center_y
                pygame.draw.line(eye_surface, self.colors['eyes']['outline'],
                               (corner_x, corner_y),
                               (corner_x + side * 5, corner_y),
                               1)
            
            # 合并到主surface
            surface.blit(eye_surface, (pos[0] - 30, pos[1] - 20))

    def draw_advanced_mouth(self, surface, mouth_position, values):
        """绘制精细的嘴巴"""
        center_x, center_y = mouth_position
        mouth_smile = values.get('mouth_smile', 0.0)
        mouth_open = values.get('mouth_open', 0.0)
        
        # 创建嘴巴层
        mouth_surface = pygame.Surface((80, 40), pygame.SRCALPHA)
        local_center = (40, 20)
        
        # 1. 嘴唇基本形状
        width = 40 + mouth_smile * 10
        height = 12 + mouth_open * 15
        
        # 上唇控制点
        upper_lip_points = [
            (40 - width/2, 20),          # 左角
            (40 - width/4, 18),          # 左控制点
            (40, 17 - mouth_smile * 3),  # 中点
            (40 + width/4, 18),          # 右控制点
            (40 + width/2, 20)           # 右角
        ]
        
        # 下唇控制点
        lower_lip_points = [
            (40 - width/2, 20),          # 左角
            (40 - width/4, 22 + mouth_open * 8),  # 左控制点
            (40, 23 + mouth_open * 10 + mouth_smile * 2),  # 中点
            (40 + width/4, 22 + mouth_open * 8),  # 右控制点
            (40 + width/2, 20)           # 右角
        ]
        
        # 2. 绘制嘴唇
        # 上唇
        upper_curve = self.bezier_curve_points(upper_lip_points)
        # 下唇
        lower_curve = self.bezier_curve_points(lower_lip_points)
        
        # 合并曲线形成完整的嘴唇形状
        lip_shape = upper_curve + lower_curve[::-1]
        
        # 3. 绘制嘴唇颜色层
        for i, color in enumerate(self.colors['lips']['base']):
            offset = i * 1
            offset_shape = [(x, y+offset) for x, y in lip_shape]
            pygame.draw.polygon(mouth_surface, (*color, 200), offset_shape)
        
        # 4. 如果嘴巴张开，绘制口腔
        if mouth_open > 0.2:
            inner_color = (120, 40, 40)
            inner_points = [
                *upper_curve[len(upper_curve)//4:3*len(upper_curve)//4],
                *lower_curve[len(lower_curve)//4:3*len(lower_curve)//4][::-1]
            ]
            pygame.draw.polygon(mouth_surface, inner_color, inner_points)
        
        # 5. 添加高光
        highlight_points = self.bezier_curve_points(upper_lip_points[:3])
        for i, color in enumerate(self.colors['lips']['highlight']):
            pygame.draw.lines(mouth_surface, (*color, 100),
                            False, highlight_points, 2-i)
        
        # 6. 嘴角阴影
        for side in [-1, 1]:
            corner_x = 40 + side * width/2
            shadow_points = [
                (corner_x, 20),
                (corner_x + side * 3, 20 + 2)
            ]
            pygame.draw.lines(mouth_surface, self.colors['lips']['shadow'][0],
                            False, shadow_points, 1)
        
        # 合并到主surface
        surface.blit(mouth_surface, (center_x - 40, center_y - 20))

    def draw_advanced_eyebrows(self, surface, brow_positions, values):
        """绘制精细的眉毛"""
        brow_raise = values.get('brow_raise', 0.0)
        brow_furrow = values.get('brow_furrow', 0.0)
        
        for i, pos in enumerate(brow_positions):
            side = 1 if i == 1 else -1  # 左右眉毛
            
            # 创建眉毛层
            brow_surface = pygame.Surface((80, 40), pygame.SRCALPHA)
            
            # 生成眉毛控制点
            control_points = [
                (20, 20 + side * brow_furrow * 5),  # 内侧点
                (40, 15 - brow_raise * 10),         # 中点
                (60, 20 + side * brow_furrow * 2)   # 外侧点
            ]
            
            # 生成眉毛形状点
            brow_points = []
            steps = 20
            
            # 上边缘
            for step in range(steps):
                t = step / (steps-1)
                point = self.bezier_point(control_points, t)
                offset = math.sin(t * math.pi) * 4  # 眉毛粗细变化
                brow_points.append((point[0], point[1] - offset))
            
            # 下边缘
            for step in range(steps-1, -1, -1):
                t = step / (steps-1)
                point = self.bezier_point(control_points, t)
                offset = math.sin(t * math.pi) * 2
                brow_points.append((point[0], point[1] + offset))
            
            # 绘制眉毛主体
            pygame.draw.polygon(brow_surface, self.colors['hair']['base'][0], brow_points)
            
            # 添加眉毛纹理
            self.add_eyebrow_texture(brow_surface, control_points)
            
            # 合并到主surface
            surface.blit(brow_surface, (pos[0] - 40, pos[1] - 20))

    def draw_eyebrow_segment(self, surface, points):
        """绘制眉毛段落"""
        # 生成眉毛轮廓点
        brow_points = []
        steps = 20
        
        # 上边缘点
        for i in range(steps):
            t = i / (steps - 1)
            point = self.bezier_point(points, t)
            offset = math.sin(t * math.pi) * 4  # 眉毛粗细变化
            brow_points.append((point[0], point[1] - offset))
        
        # 下边缘点
        for i in range(steps-1, -1, -1):
            t = i / (steps - 1)
            point = self.bezier_point(points, t)
            offset = math.sin(t * math.pi) * 2
            brow_points.append((point[0], point[1] + offset))
        
        # 绘制眉毛形状
        self.draw_smooth_polygon(surface, self.colors['hair']['base'][0], brow_points)

    def add_eyebrow_texture(self, surface, points):
        """添加眉毛纹理"""
        steps = 30
        for i in range(steps):
            t = i / (steps - 1)
            base_point = self.bezier_point(points, t)
            
            # 添加多个细小的毛发线条
            for _ in range(3):
                start_x = base_point[0] + random.uniform(-2, 2)
                start_y = base_point[1] + random.uniform(-2, 2)
                
                length = random.uniform(4, 8)
                angle = math.radians(-60 + random.uniform(-20, 20))
                
                end_x = start_x + math.cos(angle) * length
                end_y = start_y + math.sin(angle) * length
                
                color = (*self.colors['hair']['base'][0], 150)
                pygame.draw.aaline(surface, color, (start_x, start_y), (end_x, end_y))

    def draw_advanced_nose(self, surface, nose_position, values):
        """绘制精细的鼻子"""
        center_x, center_y = nose_position
        
        # 创建鼻子层
        nose_surface = pygame.Surface((40, 60), pygame.SRCALPHA)
        local_center = (20, 30)
        
        # 1. 鼻梁
        bridge_points = [
            (20, 0),     # 上点
            (16, 15),    # 左控制点
            (24, 15),    # 右控制点
            (20, 30)     # 鼻尖
        ]
        
        # 绘制鼻梁阴影
        shadow_points = self.bezier_curve_points(bridge_points)
        for i, color in enumerate(self.colors['skin']['shadow']):
            offset = i * 1
            pygame.draw.lines(nose_surface, (*color, 100),
                            False, [(x+offset, y) for x, y in shadow_points], 2-i)
        
        # 2. 鼻翼
        nostril_width = 16
        nostril_height = 8
        for side in [-1, 1]:
            # 鼻翼轮廓
            wing_points = [
                (20, 30),  # 鼻尖
                (20 + side * 8, 32),  # 控制点
                (20 + side * nostril_width/2, 34)  # 鼻翼边缘
            ]
            
            # 绘制鼻翼
            wing_curve = self.bezier_curve_points(wing_points)
            pygame.draw.lines(nose_surface, self.colors['skin']['shadow'][1],
                            False, wing_curve, 2)
        
        # 3. 鼻孔
        for side in [-1, 1]:
            nostril_center = (20 + side * 6, 35)
            # 椭圆形鼻孔
            nostril_rect = pygame.Rect(
                nostril_center[0] - 3,
                nostril_center[1] - 2,
                6, 4
            )
            self.draw_smooth_ellipse(nose_surface,
                                   (*self.colors['skin']['shadow'][2], 150),
                                   nostril_rect)
        
        # 合并到主surface
        surface.blit(nose_surface, (center_x - 20, center_y - 30))

    def draw_smooth_polygon(self, surface, color, points):
        # Implementation of draw_smooth_polygon method
        pass

    def draw_smooth_circle(self, surface, color, center, radius):
        # Implementation of draw_smooth_circle method
        pass

    def point_in_polygon(self, point, polygon):
        # Implementation of point_in_polygon method
        pass

    def add_blush(self, surface, values):
        # Implementation of add_blush method
        pass

    def add_accessories(self, surface, values):
        """添加装饰元素"""
        accessories = values.get('accessories', [])
        
        for accessory in accessories:
            if accessory == 'glasses':
                self.draw_glasses(surface, values)
            elif accessory == 'hair_clip':
                self.draw_hair_clip(surface, values)
            elif accessory == 'earrings':
                self.draw_earrings(surface, values)

    def draw_glasses(self, surface, values):
        """绘制眼镜"""
        center_x = self.width // 2
        center_y = self.height // 2 - 50
        
        # 眼镜参数
        glass_width = 100
        glass_height = 60
        bridge_width = 40
        
        # 创建眼镜层
        glasses_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        for side in [-1, 1]:
            # 镜框
            frame_rect = pygame.Rect(
                center_x + side * (glass_width/2 + bridge_width/2) - glass_width/2,
                center_y - glass_height/2,
                glass_width,
                glass_height
            )
            
            # 绘制镜片
            self.draw_smooth_ellipse(glasses_surface, (200, 200, 200, 30), frame_rect)
            
            # 绘制镜框
            frame_points = self.generate_glass_frame(frame_rect)
            for i in range(2):
                offset = i * 2
                offset_points = [(x+offset, y+offset) for x, y in frame_points]
                pygame.draw.lines(glasses_surface, (60, 60, 60), True, offset_points, 2-i)
        
        # 绘制镜架
        bridge_points = [
            (center_x - 20, center_y),
            (center_x, center_y - 5),
            (center_x + 20, center_y)
        ]
        self.draw_bezier_curve(glasses_surface, bridge_points, (60, 60, 60), 2)
        
        # 绘制镜腿
        for side in [-1, 1]:
            temple_start = (center_x + side * (glass_width/2 + bridge_width/2), center_y)
            temple_end = (temple_start[0] + side * 80, center_y + 40)
            pygame.draw.line(glasses_surface, (60, 60, 60), temple_start, temple_end, 2)
        
        surface.blit(glasses_surface, (0, 0))

    def generate_glass_frame(self, rect):
        """生成眼镜框的点"""
        points = []
        steps = 20
        
        for i in range(steps):
            t = i / steps
            angle = 2 * math.pi * t
            
            x = rect.centerx + rect.width/2 * math.cos(angle)
            y = rect.centery + rect.height/2 * math.sin(angle)
            
            points.append((int(x), int(y)))
        
        return points

    def draw_hair_clip(self, surface, values):
        """绘制发夹"""
        center_x = self.width // 2 + 100  # 位置可调整
        center_y = self.height // 2 - 100
        
        clip_surface = pygame.Surface((60, 30), pygame.SRCALPHA)
        
        # 发夹形状
        clip_points = [
            (0, 15),
            (15, 5),
            (45, 5),
            (60, 15),
            (45, 25),
            (15, 25)
        ]
        
        # 绘制发夹主体
        self.draw_smooth_polygon(clip_surface, (200, 150, 150), clip_points)
        
        # 添加装饰
        decoration_center = (30, 15)
        self.draw_smooth_circle(clip_surface, (220, 170, 170), decoration_center, 8)
        self.draw_smooth_circle(clip_surface, (240, 190, 190), decoration_center, 5)
        
        # 旋转发夹
        rotated_clip = pygame.transform.rotate(clip_surface, -30)
        surface.blit(rotated_clip, (center_x - 30, center_y - 15))

    def draw_smooth_ellipse(self, surface, ellipse, rect):
        # Implementation of draw_smooth_ellipse method
        pass

    def draw_bezier_curve(self, surface, points, color, thickness=1):
        """绘制贝塞尔曲线"""
        curve_points = self.bezier_curve_points(points)
        if len(curve_points) > 1:
            pygame.draw.lines(surface, color, False, curve_points, thickness)

    def bezier_point(self, points, t):
        """计算贝塞尔曲线上的点"""
        if len(points) == 1:
            return points[0]
        
        new_points = []
        for i in range(len(points) - 1):
            x = points[i][0] * (1 - t) + points[i + 1][0] * t
            y = points[i][1] * (1 - t) + points[i + 1][1] * t
            new_points.append((x, y))
        
        return self.bezier_point(new_points, t)

    def bezier_curve_points(self, points, steps=30):
        """生成贝塞尔曲线的点序列"""
        curve_points = []
        for i in range(steps):
            t = i / (steps - 1)
            point = self.bezier_point(points, t)
            curve_points.append((int(point[0]), int(point[1])))
        return curve_points 
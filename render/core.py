import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GL import shaders
import logging
from typing import Dict, Any, Optional
import torch

class ModelRenderer:
    """3D模型渲染器"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化OpenGL上下文
        if not glfw.init():
            raise RuntimeError("无法初始化GLFW")
            
        # 创建离屏渲染窗口
        glfw.window_hint(glfw.VISIBLE, False)
        self.window = glfw.create_window(
            800, 600, "Model Renderer", None, None)
            
        if not self.window:
            glfw.terminate()
            raise RuntimeError("无法创建GLFW窗口")
            
        glfw.make_context_current(self.window)
        
        # 初始化渲染状态
        self.shader_program = None
        self.vao = None
        self.vbo = None
        self.vertices = None
        self.weights = None
        
        self._init_gl()
        
    def _init_gl(self):
        """初始化OpenGL状态"""
        # 编译着色器
        vertex_shader = shaders.compileShader("""
            #version 330
            layout(location = 0) in vec3 position;
            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;
            
            void main() {
                gl_Position = projection * view * model * vec4(position, 1.0);
            }
        """, GL_VERTEX_SHADER)
        
        fragment_shader = shaders.compileShader("""
            #version 330
            out vec4 fragColor;
            
            void main() {
                fragColor = vec4(1.0, 1.0, 1.0, 1.0);
            }
        """, GL_FRAGMENT_SHADER)
        
        self.shader_program = shaders.compileProgram(
            vertex_shader, fragment_shader)
            
    def render_frame(self, 
                    vertices: np.ndarray,
                    weights: Optional[np.ndarray] = None) -> np.ndarray:
        """渲染一帧
        
        Args:
            vertices: 顶点坐标数组
            weights: 可选的权重数组
            
        Returns:
            渲染的帧图像
        """
        if weights is not None:
            self.weights = weights
            
        # 更新顶点数据
        if self.vao is None:
            self.vao = glGenVertexArrays(1)
            self.vbo = glGenBuffers(1)
            
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, 
                    vertices, GL_STATIC_DRAW)
                    
        # 设置顶点属性
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        
        # 清除缓冲区
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # 使用着色器程序
        glUseProgram(self.shader_program)
        
        # 渲染模型
        glDrawArrays(GL_TRIANGLES, 0, len(vertices))
        
        # 读取帧缓冲
        width, height = glfw.get_framebuffer_size(self.window)
        frame = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        frame = np.frombuffer(frame, dtype=np.uint8)
        frame = frame.reshape(height, width, 3)
        
        return frame
        
    def __del__(self):
        """清理资源"""
        if self.window:
            glfw.destroy_window(self.window)
        glfw.terminate()

    def render_pose(self, binding: SkeletonBinding):
        """渲染绑定后的模型"""
        # 1. 更新骨骼矩阵
        binding.skeleton.update_matrices()
        
        # 2. 准备顶点数据
        vertices = binding.mesh_points
        weights = binding.weights
        
        # 3. 计算蒙皮矩阵
        skinning_matrices = np.array([
            binding.skeleton.joint_matrices[i] @ bone.inverse_bind_matrix
            for i, bone in enumerate(binding.bones)
        ])
        
        # 4. 应用GPU加速变形
        deformed_vertices = self._deform_vertices_gpu(
            vertices, weights, skinning_matrices
        )
        
        # 5. 渲染网格
        self._draw_mesh(deformed_vertices)

    def _deform_vertices_gpu(self, vertices, weights, matrices):
        """使用PyTorch进行GPU加速变形"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        v_tensor = torch.tensor(vertices, dtype=torch.float32, device=device)
        w_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        m_tensor = torch.tensor(matrices, dtype=torch.float32, device=device)
        
        # 批处理矩阵乘法
        transformed = torch.einsum('bij,bj->bi', m_tensor, w_tensor)
        deformed = torch.matmul(v_tensor.unsqueeze(1), transformed).squeeze()
        
        return deformed.cpu().numpy()

    def _apply_tessellation(self, vertices):
        """应用曲面细分"""
        # 使用OpenGL曲面细分着色器
        glPatchParameteri(GL_PATCH_VERTICES, 3)
        glUseProgram(self.tessellation_shader)
        
        # 设置细分级别
        glUniform1f(glGetUniformLocation(self.tessellation_shader, "TessLevel"), 
                    self._calculate_tess_level())
        
        # 绘制细分后的网格
        glBindVertexArray(self.vao)
        glDrawElements(GL_PATCHES, self.index_count, GL_UNSIGNED_INT, None)

    def _setup_advanced_shading(self):
        """配置PBR着色器"""
        self.shader_program = self._compile_shader("""
            #version 460
            layout(location = 0) in vec3 position;
            layout(location = 1) in vec3 normal;
            layout(location = 2) in vec2 texCoord;
            
            uniform mat4 MVP;
            out vec3 vPosition;
            out vec3 vNormal;
            out vec2 vTexCoord;
            
            void main() {
                vPosition = position;
                vNormal = normalize(normal);
                vTexCoord = texCoord;
                gl_Position = MVP * vec4(position, 1.0);
            }
        """, """
            #version 460
            in vec3 vPosition;
            in vec3 vNormal;
            in vec2 vTexCoord;
            
            uniform sampler2D albedoMap;
            uniform sampler2D normalMap;
            uniform sampler2D roughnessMap;
            
            out vec4 FragColor;
            
            void main() {
                // PBR光照计算
                vec3 N = normalize(vNormal);
                vec3 V = normalize(-vPosition);
                
                // 从法线贴图获取细节法线
                vec3 detailNormal = texture(normalMap, vTexCoord).xyz * 2.0 - 1.0;
                N = normalize(N + detailNormal);
                
                // 计算光照
                vec3 lightColor = vec3(1.0);
                vec3 lightDir = normalize(vec3(1.0, 1.0, 0.5));
                
                float diff = max(dot(N, lightDir), 0.0);
                vec3 diffuse = diff * lightColor;
                
                FragColor = vec4(diffuse * texture(albedoMap, vTexCoord).rgb, 1.0);
            }
        """) 
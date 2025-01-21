import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GL import shaders
import logging
from typing import Dict, Any, Optional

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
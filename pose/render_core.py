class ModelRenderer:
    def __init__(self, config: Dict[str, Any]):
        # ... 之前的初始化代码 ...
        self.skeleton = Skeleton()
        self._init_skeleton()
        self._init_shaders()  # 更新着色器以支持骨骼动画
        
    def _init_skeleton(self):
        """初始化骨骼系统"""
        # 创建基础骨骼结构（以人形模型为例）
        joints = [
            Joint("Hips", -1, np.array([0, 0, 0]), np.zeros(3), [], np.eye(4)),
            Joint("Spine", 0, np.array([0, 0.1, 0]), np.zeros(3), [], np.eye(4)),
            Joint("Chest", 1, np.array([0, 0.2, 0]), np.zeros(3), [], np.eye(4)),
            Joint("Neck", 2, np.array([0, 0.2, 0]), np.zeros(3), [], np.eye(4)),
            Joint("Head", 3, np.array([0, 0.1, 0]), np.zeros(3), [], np.eye(4)),
            # 左臂
            Joint("LeftShoulder", 2, np.array([0.2, 0, 0]), np.zeros(3), [], np.eye(4)),
            Joint("LeftArm", 5, np.array([0.1, 0, 0]), np.zeros(3), [], np.eye(4)),
            Joint("LeftForeArm", 6, np.array([0.15, 0, 0]), np.zeros(3), [], np.eye(4)),
            Joint("LeftHand", 7, np.array([0.1, 0, 0]), np.zeros(3), [], np.eye(4)),
            # 右臂
            Joint("RightShoulder", 2, np.array([-0.2, 0, 0]), np.zeros(3), [], np.eye(4)),
            Joint("RightArm", 9, np.array([-0.1, 0, 0]), np.zeros(3), [], np.eye(4)),
            Joint("RightForeArm", 10, np.array([-0.15, 0, 0]), np.zeros(3), [], np.eye(4)),
            Joint("RightHand", 11, np.array([-0.1, 0, 0]), np.zeros(3), [], np.eye(4)),
        ]
        
        # 设置父子关系
        for i, joint in enumerate(joints):
            if joint.parent_id >= 0:
                joints[joint.parent_id].children.append(i)
                
        # 添加到骨骼系统
        for joint in joints:
            self.skeleton.add_joint(joint)
            
        self.skeleton.update_matrices()
        
    def _init_shaders(self):
        """初始化支持骨骼动画的着色器"""
        vertex_shader = """
        #version 330
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 normal;
        layout(location = 2) in vec2 texCoord;
        layout(location = 3) in ivec4 boneIds;
        layout(location = 4) in vec4 weights;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform mat4 bones[100];  // 骨骼矩阵数组
        
        out vec2 TexCoord;
        out vec3 Normal;
        
        void main() {
            mat4 boneTransform = bones[boneIds[0]] * weights[0];
            boneTransform += bones[boneIds[1]] * weights[1];
            boneTransform += bones[boneIds[2]] * weights[2];
            boneTransform += bones[boneIds[3]] * weights[3];
            
            vec4 localPosition = boneTransform * vec4(position, 1.0);
            gl_Position = projection * view * model * localPosition;
            
            TexCoord = texCoord;
            Normal = mat3(transpose(inverse(model * boneTransform))) * normal;
        }
        """
        
        # ... fragment shader 代码 ...
        
    def update_pose(self, pose_data: Dict[str, Any]):
        """更新骨骼姿态"""
        # 从姿态数据更新关节旋转
        for joint_name, rotation in pose_data.items():
            for i, joint in enumerate(self.skeleton.joints):
                if joint.name == joint_name:
                    joint.initial_rotation = rotation
                    break
                    
        # 更新骨骼矩阵
        self.skeleton.update_matrices()
        
    def render_frame(self, pose_data: Dict[str, Any]) -> np.ndarray:
        """渲染一帧"""
        self.update_pose(pose_data)
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.shader)
        
        # 传递骨骼矩阵到着色器
        for i, matrix in enumerate(self.skeleton.joint_matrices):
            location = glGetUniformLocation(self.shader, f"bones[{i}]")
            glUniformMatrix4fv(location, 1, GL_FALSE, matrix)
            
        # ... 其余渲染代码 ... 
class PoseRenderer {
    constructor(canvas) {
        if (!canvas) {
            throw new Error('Canvas is required');
        }
        
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        
        if (!this.ctx) {
            throw new Error('Failed to get canvas context');
        }
        
        // 更新样式配置
        this.styles = {
            pose: {
                keypoints: {
                    radius: 6,
                    color: '#00ff00'  // 绿色
                },
                connections: {
                    width: 3,
                    color: '#00ff00'
                }
            },
            face: {
                keypoints: {
                    radius: 2,
                    color: '#ff0000'  // 红色
                },
                connections: {
                    width: 1,
                    color: '#ff0000'
                }
            },
            hands: {
                keypoints: {
                    radius: 3,
                    color: '#ffff00'  // 黄色
                },
                connections: {
                    width: 2,
                    color: '#ffff00'
                }
            }
        };
        
        // 定义连接关系
        this.connections = {
            pose: [
                [11, 12], // 肩膀
                [11, 13], [13, 15], // 左臂
                [12, 14], [14, 16], // 右臂
                [11, 23], [23, 25], [25, 27], // 左腿
                [12, 24], [24, 26], [26, 28]  // 右腿
            ],
            face: [
                // 眉毛
                [33, 246], [246, 161], [161, 160], [160, 159],  // 左眉
                [133, 155], [155, 154], [154, 153], [153, 145],  // 右眉
                // 眼睛
                [33, 7], [7, 163], [163, 144], [144, 145],  // 左眼
                [362, 382], [382, 381], [381, 380], [380, 374],  // 右眼
                // 嘴巴
                [61, 185], [185, 40], [40, 39], [39, 37],  // 上唇
                [0, 267], [267, 269], [269, 270], [270, 409]  // 下唇
            ],
            hands: [
                [0, 1], [1, 2], [2, 3], [3, 4],  // 拇指
                [0, 5], [5, 6], [6, 7], [7, 8],  // 食指
                [0, 9], [9, 10], [10, 11], [11, 12],  // 中指
                [0, 13], [13, 14], [14, 15], [15, 16],  // 无名指
                [0, 17], [17, 18], [18, 19], [19, 20]  // 小指
            ]
        };
        
        // 设置初始画布尺寸
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
        
        console.log('PoseRenderer 初始化完成', {
            canvas: this.canvas,
            context: this.ctx,
            dimensions: {
                width: this.canvas.width,
                height: this.canvas.height
            }
        });
    }
    
    resizeCanvas() {
        const container = this.canvas.parentElement;
        if (!container) {
            console.error('Canvas container not found');
            return;
        }
        
        // 保存旧尺寸
        const oldWidth = this.canvas.width;
        const oldHeight = this.canvas.height;
        
        // 设置新尺寸
        this.canvas.width = container.clientWidth;
        this.canvas.height = container.clientHeight;
        
        console.log('画布尺寸已更新', {
            container: {
                width: container.clientWidth,
                height: container.clientHeight
            },
            canvas: {
                old: { width: oldWidth, height: oldHeight },
                new: { width: this.canvas.width, height: this.canvas.height }
            }
        });
        
        // 更新调试信息显示
        const canvasSize = document.getElementById('canvasSize');
        if (canvasSize) {
            canvasSize.textContent = `${this.canvas.width}x${this.canvas.height}`;
        }
    }
    
    drawPose(data) {
        try {
            console.log('开始绘制关键点', {
                canvasSize: {
                    width: this.canvas.width,
                    height: this.canvas.height,
                    clientWidth: this.canvas.clientWidth,
                    clientHeight: this.canvas.clientHeight
                },
                data: {
                    pose: data.pose?.length || 0,
                    face: data.face?.length || 0,
                    leftHand: data.left_hand?.length || 0,
                    rightHand: data.right_hand?.length || 0
                }
            });

            // 清空画布并填充黑色背景
            this.ctx.fillStyle = '#000000';
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
            
            if (!data) {
                console.warn('无效的关键点数据');
                return;
            }
            
            // 绘制姿态关键点
            if (data.pose && data.pose.length > 0) {
                this.drawConnections(data.pose, this.connections.pose, this.styles.pose);
                this.drawKeypoints(data.pose, this.styles.pose.keypoints, 'pose');
            }
            
            // 绘制面部关键点
            if (data.face && data.face.length > 0) {
                this.drawConnections(data.face, this.connections.face, this.styles.face);
                this.drawKeypoints(data.face, this.styles.face.keypoints, 'face');
            }
            
            // 绘制手部关键点
            if (data.left_hand && data.left_hand.length > 0) {
                this.drawConnections(data.left_hand, this.connections.hands, this.styles.hands);
                this.drawKeypoints(data.left_hand, this.styles.hands.keypoints, 'left_hand');
            }
            if (data.right_hand && data.right_hand.length > 0) {
                this.drawConnections(data.right_hand, this.connections.hands, this.styles.hands);
                this.drawKeypoints(data.right_hand, this.styles.hands.keypoints, 'right_hand');
            }
            
        } catch (error) {
            console.error('绘制关键点时出错:', error);
        }
    }
    
    drawKeypoints(landmarks, style, type) {
        landmarks.forEach((landmark, index) => {
            const x = landmark.x * this.canvas.width;
            const y = landmark.y * this.canvas.height;
            
            console.log(`绘制${type}关键点 ${index}:`, {
                original: { x: landmark.x, y: landmark.y },
                scaled: { x, y }
            });
            
            this.ctx.beginPath();
            this.ctx.arc(x, y, style.radius, 0, 2 * Math.PI);
            this.ctx.fillStyle = style.color;
            this.ctx.fill();
        });
    }
    
    drawConnections(landmarks, connections, style) {
        this.ctx.strokeStyle = style.connections.color;
        this.ctx.lineWidth = style.connections.width;
        
        connections.forEach(([i, j]) => {
            if (landmarks[i] && landmarks[j]) {
                const x1 = landmarks[i].x * this.canvas.width;
                const y1 = landmarks[i].y * this.canvas.height;
                const x2 = landmarks[j].x * this.canvas.width;
                const y2 = landmarks[j].y * this.canvas.height;
                
                this.ctx.beginPath();
                this.ctx.moveTo(x1, y1);
                this.ctx.lineTo(x2, y2);
                this.ctx.stroke();
            }
        });
    }
} 
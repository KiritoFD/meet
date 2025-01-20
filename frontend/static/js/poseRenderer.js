class PoseRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.landmarks = [];
        
        // 设置线条样式
        this.styles = {
            pose: {
                color: '#00ff00',
                width: 2
            },
            face: {
                color: '#ff0000',
                width: 1
            },
            hands: {
                color: '#0000ff',
                width: 1.5
            }
        };
        
        // 定义身体部位的连接关系
        this.connections = {
            pose: [
                // 躯干
                [11, 12], // 肩膀
                [11, 23], [12, 24], // 躯干两侧
                [23, 24], // 臀部
                
                // 左臂
                [11, 13], [13, 15],
                
                // 右臂
                [12, 14], [14, 16],
                
                // 左腿
                [23, 25], [25, 27], [27, 31],
                
                // 右腿
                [24, 26], [26, 28], [28, 32]
            ],
            face: [
                // 面部轮廓
                [10, 338], [338, 297], [297, 332], [332, 284], [284, 251], [251, 389],
                [389, 356], [356, 454], [454, 323], [323, 361], [361, 288], [288, 397],
                [397, 365], [365, 379], [379, 378], [378, 400], [400, 377], [377, 152],
                [152, 148], [148, 176], [176, 149], [149, 150], [150, 136], [136, 172],
                [172, 58], [58, 132], [132, 93], [93, 234], [234, 127], [127, 162],
                [162, 21], [21, 54], [54, 103], [103, 67], [67, 109], [109, 10]
            ]
        };
    }
    
    updatePose(poseData) {
        this.landmarks = poseData.landmarks;
        this.render();
    }
    
    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // 绘制关键点
        this.landmarks.forEach(landmark => {
            const x = landmark.x * this.canvas.width;
            const y = landmark.y * this.canvas.height;
            
            this.ctx.beginPath();
            this.ctx.arc(x, y, 3, 0, 2 * Math.PI);
            this.ctx.fillStyle = `rgba(255, 0, 0, ${landmark.visibility})`;
            this.ctx.fill();
        });
        
        // 绘制连接线
        this.drawConnections();
    }
    
    drawConnections() {
        // 定义关键点连接关系
        const connections = [
            [11, 12], // 肩膀
            [11, 13], [13, 15], // 左臂
            [12, 14], [14, 16], // 右臂
            [11, 23], [23, 25], [25, 27], // 左腿
            [12, 24], [24, 26], [26, 28]  // 右腿
        ];
        
        connections.forEach(([start, end]) => {
            const startPoint = this.landmarks[start];
            const endPoint = this.landmarks[end];
            
            if (startPoint && endPoint) {
                this.ctx.beginPath();
                this.ctx.moveTo(
                    startPoint.x * this.canvas.width,
                    startPoint.y * this.canvas.height
                );
                this.ctx.lineTo(
                    endPoint.x * this.canvas.width,
                    endPoint.y * this.canvas.height
                );
                this.ctx.strokeStyle = '#00ff00';
                this.ctx.lineWidth = 2;
                this.ctx.stroke();
            }
        });
    }
    
    drawPose(poseData) {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        if (!poseData) return;
        
        // 绘制身体姿态
        if (poseData.pose) {
            this.ctx.strokeStyle = this.styles.pose.color;
            this.ctx.lineWidth = this.styles.pose.width;
            this.drawConnections(poseData.pose, this.connections.pose);
        }
        
        // 绘制面部网格
        if (poseData.face) {
            this.ctx.strokeStyle = this.styles.face.color;
            this.ctx.lineWidth = this.styles.face.width;
            this.drawConnections(poseData.face, this.connections.face);
        }
        
        // 绘制手部
        if (poseData.left_hand || poseData.right_hand) {
            this.ctx.strokeStyle = this.styles.hands.color;
            this.ctx.lineWidth = this.styles.hands.width;
            
            if (poseData.left_hand) {
                this.drawHandLandmarks(poseData.left_hand);
            }
            if (poseData.right_hand) {
                this.drawHandLandmarks(poseData.right_hand);
            }
        }
    }
    
    drawHandLandmarks(landmarks) {
        // 手部关键点连接
        for (let i = 0; i < landmarks.length - 1; i++) {
            const current = this.transformPoint(landmarks[i]);
            const next = this.transformPoint(landmarks[i + 1]);
            
            this.ctx.beginPath();
            this.ctx.moveTo(current.x, current.y);
            this.ctx.lineTo(next.x, next.y);
            this.ctx.stroke();
        }
    }
    
    transformPoint(point) {
        return {
            x: point.x * this.canvas.width,
            y: point.y * this.canvas.height
        };
    }
} 
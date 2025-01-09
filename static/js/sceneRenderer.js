class SceneRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.initialImage = null;
        this.lastPoseData = null;
        this.lastTransform = null;
        this.smoothFactor = 0.9;
        
        // 绑定方法
        this.drawFrame = this.drawFrame.bind(this);
    }

    setInitialFrame(imageData) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                console.log('初始帧加载成功:', img.width, 'x', img.height);
                this.initialImage = img;
                this.drawFrame();
                resolve();
            };
            img.onerror = (error) => {
                console.error('初始帧加载失败:', error);
                reject(error);
            };
            img.src = 'data:image/jpeg;base64,' + imageData;
        });
    }

    updatePose(poseData) {
        this.lastPoseData = poseData;
        requestAnimationFrame(this.drawFrame);
    }

    drawFrame() {
        if (!this.initialImage || !this.lastPoseData) {
            return;
        }

        try {
            // 清空画布
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            
            // 计算变换
            const transform = this.calculateTransform();
            if (!transform) {
                this.drawOriginalImage();
                return;
            }

            // 应用变换并绘制
            this.ctx.save();
            this.ctx.translate(this.canvas.width/2, this.canvas.height/2);
            this.ctx.transform(
                transform.a, transform.b,
                transform.c, transform.d,
                transform.e, transform.f
            );
            
            // 绘制变换后的图像
            this.ctx.drawImage(
                this.initialImage,
                -this.canvas.width/2, -this.canvas.height/2,
                this.canvas.width, this.canvas.height
            );
            
            this.ctx.restore();
        } catch (error) {
            console.error('绘制错误:', error);
            this.drawOriginalImage();
        }
    }

    calculateTransform() {
        try {
            const { head, neck, leftShoulder, rightShoulder } = this.extractKeyPoints();
            if (!head || !neck || !leftShoulder || !rightShoulder) {
                return null;
            }

            // 计算基本变换参数
            const centerPoint = this.calculateCenter(leftShoulder, rightShoulder);
            const scale = this.calculateScale(leftShoulder, rightShoulder);
            const angle = this.calculateAngle(leftShoulder, rightShoulder);
            const translation = this.calculateTranslation(centerPoint);

            // 构建变换矩阵
            return this.buildTransformMatrix(angle, scale, translation);
        } catch (error) {
            console.error('计算变换失败:', error);
            return null;
        }
    }

    extractKeyPoints() {
        return {
            head: this.lastPoseData[0],
            neck: this.lastPoseData[1],
            leftShoulder: this.lastPoseData[11],
            rightShoulder: this.lastPoseData[12]
        };
    }

    calculateCenter(leftShoulder, rightShoulder) {
        return {
            x: (leftShoulder[0] + rightShoulder[0]) / 2,
            y: (leftShoulder[1] + rightShoulder[1]) / 2
        };
    }

    calculateScale(leftShoulder, rightShoulder) {
        const shoulderWidth = Math.hypot(
            rightShoulder[0] - leftShoulder[0],
            rightShoulder[1] - leftShoulder[1]
        );
        return shoulderWidth / (this.canvas.width * 0.3);
    }

    calculateAngle(leftShoulder, rightShoulder) {
        return Math.atan2(
            rightShoulder[1] - leftShoulder[1],
            rightShoulder[0] - leftShoulder[0]
        );
    }

    calculateTranslation(centerPoint) {
        return {
            dx: this.canvas.width/2 - centerPoint.x,
            dy: this.canvas.height/2 - centerPoint.y
        };
    }

    buildTransformMatrix(angle, scale, translation) {
        const transform = {
            a: Math.cos(angle) * scale,
            b: Math.sin(angle) * scale,
            c: -Math.sin(angle) * scale,
            d: Math.cos(angle) * scale,
            e: translation.dx,
            f: translation.dy
        };

        // 应用平滑处理
        if (this.lastTransform) {
            return this.smoothTransform(transform);
        }

        this.lastTransform = transform;
        return transform;
    }

    smoothTransform(newTransform) {
        const smoothed = {};
        for (const key in newTransform) {
            smoothed[key] = this.smoothValue(
                this.lastTransform[key],
                newTransform[key]
            );
        }
        this.lastTransform = smoothed;
        return smoothed;
    }

    smoothValue(current, target) {
        return current * (1 - this.smoothFactor) + target * this.smoothFactor;
    }

    drawOriginalImage() {
        if (this.initialImage) {
            this.ctx.drawImage(
                this.initialImage,
                0, 0,
                this.canvas.width, this.canvas.height
            );
        }
    }

    showMessage(message, isError = false) {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.fillStyle = isError ? 'red' : 'white';
        this.ctx.font = '20px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(message, this.canvas.width/2, this.canvas.height/2);
    }
} 
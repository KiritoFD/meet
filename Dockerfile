FROM python:3.7-slim

# TODO: 多阶段构建
# TODO: 添加健康检查
# TODO: 优化镜像大小
# TODO: 添加非root用户
# TODO: 添加缓存清理

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p static/img/uploads logs

# 设置环境变量
ENV PYTHONPATH=/app
ENV FLASK_APP=run.py

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["python", "run.py"] 
# 使用通用基础镜像
FROM ubuntu:latest

# 设置工作目录
WORKDIR /app

# 复制打包好的环境
COPY ./pytorch-12.4.tar.gz /tmp/env.tar.gz

# 解压环境并设置
RUN mkdir /opt/conda && \
    tar -xzf /tmp/env.tar.gz -C /opt/conda && \
    rm /tmp/env.tar.gz

# 初始化环境
SHELL ["/bin/bash", "-c"]
RUN source /opt/conda/bin/activate

# 复制项目文件
COPY . /app/

# 设置环境变量
ENV PATH /opt/conda/bin:$PATH
ENV FLASK_APP=run.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["python", "run.py"] 
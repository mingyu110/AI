FROM python:3.10-slim AS builder

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 创建虚拟环境
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 复制并安装需求文件
COPY requirements.txt .

# 分步安装依赖以提高构建缓存效率
RUN pip install --no-cache-dir numpy==1.21.6
RUN pip install --no-cache-dir scipy==1.7.3 scikit-learn==1.0.2
RUN pip install --no-cache-dir flask==2.0.3 gunicorn==20.1.0
RUN pip install --no-cache-dir boto3==1.24.59 cloudpickle==2.2.1
RUN pip install --no-cache-dir sagemaker-inference==1.10.1
RUN pip install --no-cache-dir dlr==1.10.0

# 第二阶段：创建精简的最终镜像
FROM python:3.10-slim

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_PATH="/opt/ml/model"

# 复制虚拟环境
COPY --from=builder /opt/venv /opt/venv

# 设置工作目录
WORKDIR /opt/program

# 复制推理代码
COPY inference.py .
COPY serve.py .

# 创建必要的目录并设置权限
RUN chmod +x serve.py && \
    mkdir -p /opt/ml/model && \
    chmod -R o+rwX /opt/ml && \
    python -c "import dlr; print(f'DLR版本: {dlr.__version__}')"

# 接收Neo模型路径参数
ARG NEO_MODEL_PATH=""
# 复制Neo编译优化的模型(如果提供)
COPY ${NEO_MODEL_PATH}/ /opt/ml/model/
# 显示模型目录内容
RUN ls -la /opt/ml/model/ || echo "模型目录为空，将在运行时挂载模型"

# 容器启动命令
ENTRYPOINT ["python", "/opt/program/serve.py"]

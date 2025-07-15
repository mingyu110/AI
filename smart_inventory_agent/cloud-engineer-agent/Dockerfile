FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    curl \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# 首先复制requirements文件以便更好地缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# 查找uv的安装位置并在标准PATH位置创建符号链接
RUN find / -name uv -type f 2>/dev/null | head -1 | xargs -I{} ln -sf {} /usr/local/bin/uv && \
    find / -name uvx -type f 2>/dev/null | head -1 | xargs -I{} ln -sf {} /usr/local/bin/uvx

# 现在使用uv安装Python
RUN uv python install 3.11 3.12
RUN pip install --no-cache-dir diagrams

# 复制应用代码
COPY . .

# 暴露Streamlit端口
EXPOSE 8501

# 设置环境变量
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# 运行应用的命令
CMD ["streamlit", "run", "app.py"]

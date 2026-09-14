FROM python:3.11-slim

WORKDIR /app

# 换国内 apt 镜像源（阿里云），加速系统依赖安装
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

RUN pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/ && \
    pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt

# 复制项目代码
COPY . .

# 暴露端口（统一 9003）
EXPOSE 9003

# 启动服务（端口可用 PORT 环境变量覆盖）
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-9003}"]

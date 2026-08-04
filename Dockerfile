# 1. 基础镜像：修改为 Python 3.12 的轻量级 slim 版本
FROM python:3.12-slim

# 2. 环境变量优化 (保持不变，对 AI Harness 和算法调试极其重要)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LANG=C.UTF-8

# 3. 设置容器内的工作目录
WORKDIR /app

# 4. 安装系统级依赖 (可选)
# Python 3.12 的 slim 镜像非常精简，如果你的算法库(如 numpy, scipy)需要编译 C 扩展，取消下面这行的注释
# RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential && rm -rf /var/lib/apt/lists/*

# 5. 复制依赖文件并安装
# 确保 pip 是最新版本，以更好地支持 Python 3.12 的新特性
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. 复制项目的所有源代码到容器中
COPY . .

# 7. 默认启动命令 (根据你的实际入口文件修改)
CMD ["python", "main.py"] 
# 或者 CMD ["/bin/bash", "run_harness.sh"]
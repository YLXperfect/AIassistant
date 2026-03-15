# ---- Dockerfile ----
# 使用官方 Python 3.11 轻量镜像作为基础
FROM python:3.11-slim

# 设置工作目录（容器内的路径）
WORKDIR /app

# 设置环境变量（确保 Python 输出直接打印到终端，便于查看日志）
ENV PYTHONUNBUFFERED=1

# 将 requirements.txt 复制到容器，先安装依赖（利用 Docker 缓存）
COPY requirements_new.txt .
RUN pip install --no-cache-dir -r requirements_new.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目代码到容器
# 注意：这里根据你的实际目录结构调整
# 当前项目下 FastAPI 在 src/api 中，因此只需复制 src 和 knowledge_base
COPY ./knowledge_base /app/knowledge_base
COPY ./src /app/src

# 暴露 FastAPI 默认端口
EXPOSE 8000

# 设置容器启动时执行的命令
# 当前 FastAPI 应用在 src/api/main.py 中，实例名为 app
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
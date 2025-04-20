# 使用 Python 基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制 Python 脚本到容器内
COPY main.py .

# 安装 cron 和必要的依赖
RUN apt-get update && apt-get install -y cron
RUN pip install python-binance pandas

# 添加 cron 任务，每小时执行一次
RUN echo "0 * * * * python /app/main.py >> /var/log/cron.log 2>&1" > /etc/cron.d/my-cron-job

# 赋予 cron 任务执行权限
RUN chmod 0644 /etc/cron.d/my-cron-job

# 创建日志文件
RUN touch /var/log/cron.log

# 创建保存数据的目录
RUN mkdir -p /data

# 启动 cron
CMD cron && tail -f /var/log/cron.log    
# 使用 Python 基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制 Python 脚本到容器内
COPY main.py .
COPY telegram_log_handler.py .

# 复制初始化脚本到容器内
COPY init.sh .

# 给初始化脚本添加执行权限
RUN chmod +x init.sh

# 安装 cron 和必要的依赖
RUN apt-get update && apt-get install -y cron
RUN pip install python-binance pandas

# 设置时区，这里以亚洲上海时区为例，可按需修改
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 添加 cron 任务，每小时执行一次，将日志输出到标准输出
RUN echo "0 * * * * python /app/main.py >> /dev/stdout 2>&1" > /etc/cron.d/my-cron-job

# 赋予 cron 任务执行权限
RUN chmod 0644 /etc/cron.d/my-cron-job

# 创建保存数据的目录
RUN mkdir -p /data

# 前台运行 cron
CMD ["./init.sh"]  
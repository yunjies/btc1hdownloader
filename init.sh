#!/bin/bash

# 立即执行 Python 脚本
python main.py >> /dev/stdout 2>&1

# 启动 cron 服务
cron -f    
# 构建 Docker 镜像
docker build -t btc1hdownloader .

# 设置环境变量并运行 Docker 容器，同时进行卷映射
docker run -d \
  -e BINANCE_API_KEY='YOUR_API_KEY' \
  -e BINANCE_API_SECRET='YOUR_API_SECRET' \
  -v $(pwd)/logs:/var/log \
  -v $(pwd)/data:/data \
  btc1hdownloader    
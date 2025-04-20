import os
from binance.client import Client
import pandas as pd
import datetime

# 从环境变量获取 API 密钥和秘钥
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

if not api_key or not api_secret:
    raise ValueError("请设置 BINANCE_API_KEY 和 BINANCE_API_SECRET 环境变量。")

client = Client(api_key, api_secret)

data_path = '/data/btc_usdt_data.csv'

# 检查是否已有数据文件
if os.path.exists(data_path):
    # 读取已有数据
    existing_df = pd.read_csv(data_path)
    # 获取最后一条记录的时间戳
    last_timestamp = pd.to_datetime(existing_df['timestamp'].iloc[-1])
    # 将时间戳转换为适合币安 API 的格式
    start_time = last_timestamp.strftime('%Y-%m-%d %H:%M:%S')
else:
    # 如果没有数据文件，从一个月前开始拉取
    start_time = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')

# 获取比特币对 USDT 的 K 线数据，从上次记录时间开始
klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1DAY, start_time)

# 将新数据转换为 DataFrame
new_df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

# 转换时间戳
new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')

if os.path.exists(data_path):
    # 合并新旧数据
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    # 去除重复记录
    combined_df = combined_df.drop_duplicates(subset='timestamp')
else:
    combined_df = new_df

# 保存合并后的数据到 /data 目录下
combined_df.to_csv(data_path, index=False)
print(f"数据已保存到 {data_path}")
    
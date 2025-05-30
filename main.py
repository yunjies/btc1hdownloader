import os, sys
import logging
from binance.client import Client
import pandas as pd
import datetime
from telegram_log_handler import TelegramHandler
from feishu_log_handler import FeishuHandler
from prediction import predict_close_prices
import json
import numpy as np

# 自定义 Telegram 日志过滤器
class RemoteMessageFilter(logging.Filter):
    def filter(self, record):
        # 这里可以根据需求修改过滤规则
        # 示例：只允许包含 'error' 或 '重要' 关键字的日志通过
        keywords = ['【通知】']
        message = record.getMessage().lower()
        return any(keyword in message for keyword in keywords)

def InitLogging():
    # Telegram 配置
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

    # 飞书 配置
    FEISHU_APP_ID = os.getenv('FEISHU_APP_ID')
    FEISHU_APP_SECRET = os.getenv('FEISHU_APP_SECRET')

    # 配置日志输出到标准输出和 Telegram 和飞书
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        telegram_handler = TelegramHandler(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        telegram_handler.addFilter(RemoteMessageFilter())
        logging.getLogger().addHandler(telegram_handler)

    if FEISHU_APP_ID and FEISHU_APP_SECRET:
        feishu_handler = FeishuHandler(FEISHU_APP_ID, FEISHU_APP_SECRET)
        feishu_handler.addFilter(RemoteMessageFilter())
        logging.getLogger().addHandler(feishu_handler)

def InitBinanceClient():
    # 从环境变量获取 API 密钥和秘钥
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')

    if not api_key or not api_secret:
        logging.error("【通知】请设置 BINANCE_API_KEY 和 BINANCE_API_SECRET 环境变量。")
        raise ValueError("请设置 BINANCE_API_KEY 和 BINANCE_API_SECRET 环境变量。")
    
    client = Client(api_key, api_secret)
    return client

def PullData(client, data_path):
    # 检查是否已有数据文件
    if os.path.exists(data_path):
        # 读取已有数据
        existing_df = pd.read_csv(data_path)
        # 获取最后一条记录的时间戳
        last_timestamp = pd.to_datetime(existing_df['timestamp'].iloc[-1])
        # 将时间戳转换为适合币安 API 的格式
        start_time = last_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f"从 {start_time} 开始增量拉取数据")
    else:
        # 如果缓存文件不存在，从2012-01-01开始
        start_time = 1325376000000
        logging.info(f"未找到历史数据，从2012-01-01开始下载数据。")

    # 获取比特币对 USDT 的 K 线数据，从上次记录时间开始
    try:
        klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1HOUR, start_time)
        logging.info(f"成功拉取 {len(klines)} 条 K 线数据")
    except Exception as e:
        logging.error(f"【通知】拉取数据时发生错误: {e}")
        raise

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
    try:
        combined_df.to_csv(data_path, index=False)
        logging.info(f"数据已保存到 {data_path}")
    except Exception as e:
        logging.error(f"【通知】存数据时发生错误: {e}")
        raise    

    return combined_df

def Predict(combined_df):
    predict_result = prediction.do_prediction(combined_df)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"【通知】预测结果 {current_time}:\n{predict_result}")

if __name__ == '__main__':
    data_path = '/data/btc_usdt_data.csv'
    if len(sys.argv) > 1:
        data_path = sys.argv[1]

    InitLogging()

    # 添加调试信息
    logging.info("开始执行 Python 脚本")

    client = InitBinanceClient()

    combined_df = PullData(client, data_path)

    hours = 3
    future_close_prices = predict_close_prices(combined_df, hours, force_retrain=True)
    logging.info(f"【通知】未来 {hours} 小时的收盘价预测:\n{' '.join([str(p) for p in future_close_prices])}")
    

    

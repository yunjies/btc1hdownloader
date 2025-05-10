import os

os.environ['TELEGRAM_BOT_TOKEN'] = '1780647729:AAFt8KyFVJS2XWG-MrovhAfSlV6A2IxbwWU'
os.environ['TELEGRAM_CHAT_ID'] = '1775725405'
os.environ['FEISHU_APP_ID'] = 'cli_a88cc91f0c39d00d'
os.environ['FEISHU_APP_SECRET'] = 'QJRUntNlMI3K73fZujEgAhrXk5875Gqo'
os.environ['BINANCE_API_KEY'] = 'vQqTUaPJSHflSXpvZU9lokSy779TyYU3nyDxLDsXmBA8ene0nqq7Br7AKnUap6Ri'
os.environ['BINANCE_API_SECRET'] = 'GOGvSw5WZl8eHZp2Gs8WPwGVkdcjq7Dkes5SbAK0xVTCsOf0HLhycoLbvW3QEeBw'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.system('python main.py ./data/btc_usdt_data.csv')
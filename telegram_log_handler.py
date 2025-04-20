import logging
import requests

class TelegramHandler(logging.Handler):
    def __init__(self, bot_token, chat_id):
        super().__init__()
        self.bot_token = bot_token
        self.chat_id = chat_id

    def emit(self, record):
        log_entry = self.format(record)
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        data = {
            "chat_id": self.chat_id,
            "text": log_entry
        }
        try:
            requests.post(url, data=data)
        except Exception as e:
            print(f"Failed to send log to Telegram: {e}")    
import requests
import json
import logging

def send_message_to_feishu(app_id, app_secret, message):
    # 获取访问令牌的API地址
    token_url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal/"
    # 发送请求获取访问令牌
    response = requests.post(token_url, json={"app_id": app_id, "app_secret": app_secret})
    access_token = response.json().get("tenant_access_token")

    # 发送消息的API地址
    message_url = "https://open.feishu.cn/open-apis/im/v1/messages"
    # 设置请求头
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {access_token}"
    }
    # 设置请求参数
    params = {
        "receive_id_type": "chat_id"
    }
    # 设置请求数据
    data = {
        "receive_id": "oc_978e548c0a2bd0f8a27de6ab19626518",
        "msg_type": "text",
        "content": json.dumps({
            "text": message
        })
    }
    # 发送消息
    response = requests.post(message_url, headers=headers, params=params, json=data)
    return response.json()

class FeishuHandler(logging.Handler):
    def __init__(self, app_id, app_secret):
        super().__init__()
        self.app_id = app_id
        self.app_secret = app_secret

    def emit(self, record):
        try:
            message = self.format(record)
            send_message_to_feishu(self.app_id, self.app_secret, message)
        except Exception as e:
            self.handleError(record)
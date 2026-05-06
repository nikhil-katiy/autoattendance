import requests

BOT_TOKEN = "8698594781:AAEnHNkMzvBYBC0HNdDMG3lhc08b_8Snejs"
CHAT_ID = "-1003840626341"

def send_telegram_message(message):

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }

    requests.post(url, data=payload)
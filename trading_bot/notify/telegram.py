import os
import requests
from typing import Optional


def send_telegram_message(text: str, token: Optional[str] = None, chat_id: Optional[str] = None) -> None:
    token = token or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text[:4000], "parse_mode": "HTML"}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception:
        pass
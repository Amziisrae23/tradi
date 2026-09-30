import os
import json
import logging
import requests
from typing import Optional, Dict, Any, List
from config.settings import settings
from src.patterns.types import TradeSetup

logger = logging.getLogger("TelegramNotifier")

class TelegramNotifier:
    """Envía señales enriquecidas con botones interactivos de 1-Clic a Telegram."""

    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        self.token = token or settings.TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or settings.TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.last_update_id = 0

    def is_configured(self) -> bool:
        return bool(self.token and self.chat_id)

    def _build_trade_keyboard(self, setup: Optional[TradeSetup] = None) -> Dict[str, Any]:
        """Construye botones interactivos de 1-Clic para ejecutar o descartar la señal."""
        if not setup:
            return {}

        callback_data = f"exec:{setup.symbol}:{setup.direction}:{setup.entry_price}:{setup.stop_loss}:{setup.tp2}"
        # Telegram limita callback_data a 64 bytes
        if len(callback_data) > 64:
            callback_data = f"ex:{setup.symbol}:{setup.direction[:1]}:{setup.entry_price}:{setup.stop_loss}"

        keyboard = {
            "inline_keyboard": [
                [
                    {
                        "text": f"🟢 EJECUTAR EN BITUNIX (2% Kelly / ${settings.FIXED_RISK_USD:.0f} USD)",
                        "callback_data": callback_data
                    }
                ],
                [
                    {
                        "text": "❌ DESCARTAR SEÑAL",
                        "callback_data": "discard"
                    }
                ]
            ]
        }
        return keyboard

    def send_message(self, text: str, reply_markup: Optional[Dict[str, Any]] = None) -> bool:
        if not self.is_configured():
            return False

        url = f"{self.base_url}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text}
        if reply_markup:
            payload["reply_markup"] = reply_markup

        try:
            res = requests.post(url, json=payload, timeout=10)
            res.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Error enviando mensaje a Telegram: {e}")
            return False

    def send_photo(self, photo_path: str, caption: Optional[str] = None, reply_markup: Optional[Dict[str, Any]] = None) -> bool:
        if not self.is_configured() or not os.path.exists(photo_path):
            return False

        url = f"{self.base_url}/sendPhoto"
        short_caption = caption[:1000] + "..." if caption and len(caption) > 1000 else caption

        try:
            with open(photo_path, "rb") as photo_file:
                files = {"photo": photo_file}
                data = {"chat_id": self.chat_id}
                if short_caption:
                    data["caption"] = short_caption
                if reply_markup:
                    data["reply_markup"] = json.dumps(reply_markup)

                res = requests.post(url, data=data, files=files, timeout=20)
                res.raise_for_status()
                return True
        except Exception as e:
            logger.error(f"Error enviando foto a Telegram: {e}")
            return False

    def send_signal(self, signal_text: str, chart_path: Optional[str] = None, setup: Optional[TradeSetup] = None) -> bool:
        """Envía la señal a Telegram acompañada de los botones de ejecución de 1-Clic."""
        keyboard = self._build_trade_keyboard(setup)
        if chart_path and os.path.exists(chart_path):
            return self.send_photo(chart_path, caption=signal_text, reply_markup=keyboard)
        else:
            return self.send_message(signal_text, reply_markup=keyboard)

    def check_all_updates(self) -> List[Dict[str, Any]]:
        """
        Consulta las actualizaciones de Telegram para detectar tanto:
        1. Clics en botones interactivos (callback_query)
        2. Comandos de texto (/balance, /stats, /mode, /pause, /resume, /help)
        """
        if not self.is_configured():
            return []

        url = f"{self.base_url}/getUpdates"
        params = {"offset": self.last_update_id + 1, "timeout": 2}

        try:
            res = requests.get(url, params=params, timeout=5)
            if res.status_code == 200:
                data = res.json()
                updates = data.get("result", [])
                actions = []
                for u in updates:
                    self.last_update_id = u["update_id"]
                    # 1. Clic en botón interactivo
                    if "callback_query" in u:
                        cb = u["callback_query"]
                        cb_id = cb["id"]
                        cb_data = cb.get("data", "")
                        self.answer_callback_query(cb_id, "Procesando orden...")
                        actions.append({
                            "type": "callback",
                            "data": cb_data,
                            "user": cb.get("from", {}).get("first_name", "")
                        })
                    # 2. Mensaje de texto (Comando)
                    elif "message" in u and "text" in u["message"]:
                        msg = u["message"]
                        text = msg.get("text", "").strip()
                        actions.append({
                            "type": "command",
                            "text": text,
                            "user": msg.get("from", {}).get("first_name", "")
                        })
                return actions
        except Exception as e:
            logger.debug(f"Error verificando actualizaciones de Telegram: {e}")
        return []

    def check_button_clicks(self) -> List[Dict[str, Any]]:
        """Método de compatibilidad con clics de botones."""
        updates = self.check_all_updates()
        return [{"data": u["data"], "user": u["user"]} for u in updates if u.get("type") == "callback"]

    def answer_callback_query(self, callback_query_id: str, text: str):
        url = f"{self.base_url}/answerCallbackQuery"
        try:
            requests.post(url, json={"callback_query_id": callback_query_id, "text": text}, timeout=4)
        except Exception:
            pass


import os
import logging
import requests
from typing import Optional
from config.settings import settings

logger = logging.getLogger("TelegramNotifier")

class TelegramNotifier:
    """Envía señales y gráficos generados a un canal o chat privado de Telegram."""

    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        self.token = token or settings.TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or settings.TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.token}"

    def is_configured(self) -> bool:
        """Verifica si las credenciales de Telegram están definidas."""
        return bool(self.token and self.chat_id)

    def send_message(self, text: str) -> bool:
        """Envía un mensaje de texto formateado a Telegram."""
        if not self.is_configured():
            logger.warning("Telegram no está configurado (falta TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID en .env).")
            return False

        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text
        }

        try:
            res = requests.post(url, json=payload, timeout=10)
            res.raise_for_status()
            logger.info("✔ Mensaje enviado a Telegram con éxito.")
            return True
        except Exception as e:
            logger.error(f"Error al enviar mensaje a Telegram: {e}")
            return False

    def send_photo(self, photo_path: str, caption: Optional[str] = None) -> bool:
        """Envía una imagen (gráfico de TradingView) con texto explicativo a Telegram."""
        if not self.is_configured():
            logger.warning("Telegram no está configurado.")
            return False

        if not os.path.exists(photo_path):
            logger.error(f"El archivo de imagen no existe: {photo_path}")
            return False

        url = f"{self.base_url}/sendPhoto"
        
        # Telegram permite máximo 1024 caracteres en el caption de sendPhoto
        short_caption = caption[:1000] + "..." if caption and len(caption) > 1000 else caption

        try:
            with open(photo_path, "rb") as photo_file:
                files = {"photo": photo_file}
                data = {"chat_id": self.chat_id}
                if short_caption:
                    data["caption"] = short_caption

                res = requests.post(url, data=data, files=files, timeout=20)
                res.raise_for_status()
                logger.info("✔ Gráfico enviado a Telegram con éxito.")

                # Si el texto era más largo de 1000 caracteres, enviar el resto en un mensaje de texto adicional
                if caption and len(caption) > 1000:
                    self.send_message(caption)

                return True
        except Exception as e:
            logger.error(f"Error al enviar gráfico a Telegram: {e}")
            return False

    def send_signal(self, signal_text: str, chart_path: Optional[str] = None) -> bool:
        """Envía la señal completa con su gráfico correspondiente."""
        if chart_path and os.path.exists(chart_path):
            return self.send_photo(chart_path, caption=signal_text)
        else:
            return self.send_message(signal_text)

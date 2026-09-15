import asyncio
import orjson
import websockets
import time
import logging
from typing import Callable, Optional, List
from config.settings import settings

logger = logging.getLogger("BitunixWS")

class BitunixWebSocketClient:
    """Cliente WebSocket Asíncrono de baja latencia para Bitunix."""

    def __init__(self, symbols: Optional[List[str]] = None, interval: str = "15m"):
        self.uri = settings.BITUNIX_WS_URL
        self.symbols = [s.replace("/", "").replace("-", "").upper() for s in (symbols or settings.DEFAULT_SYMBOLS)]
        self.interval = interval
        self.ws = None
        self.is_running = False

    async def start(self, on_message_callback: Callable):
        """Inicia la conexión WebSocket y el loop de procesamiento de eventos."""
        self.is_running = True
        retry_delay = 2

        while self.is_running:
            try:
                logger.info(f"Conectando a WebSocket de Bitunix ({self.uri})...")
                async with websockets.connect(self.uri, ping_interval=None, close_timeout=5) as ws:
                    self.ws = ws
                    logger.info("🟢 Conexión WebSocket Bitunix establecida exitosamente.")
                    retry_delay = 2

                    # Suscripción a canales
                    args = []
                    for sym in self.symbols:
                        args.append({"ch": "kline", "symbol": sym, "interval": self.interval})
                        args.append({"ch": "ticker", "symbol": sym})

                    subscribe_payload = {
                        "op": "subscribe",
                        "args": args
                    }
                    await ws.send(orjson.dumps(subscribe_payload).decode("utf-8"))
                    logger.info(f"Suscrito a canales para símbolos: {self.symbols} ({self.interval})")

                    # Iniciar loop de Heartbeat en paralelo
                    heartbeat_task = asyncio.create_task(self._heartbeat_loop())

                    try:
                        async for raw_message in ws:
                            if not self.is_running:
                                break
                            try:
                                data = orjson.loads(raw_message)
                                # Ignorar pongs de respuesta
                                if data.get("op") == "ping" or "pong" in data:
                                    continue
                                
                                # Procesar mensaje
                                await on_message_callback(data)
                            except Exception as parse_err:
                                logger.debug(f"Error parseando mensaje WS: {parse_err}")
                    finally:
                        heartbeat_task.cancel()

            except (websockets.ConnectionClosed, websockets.WebSocketException, Exception) as e:
                logger.warning(f"Desconexión WebSocket Bitunix: {e}. Reconectando en {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30)

    async def _heartbeat_loop(self):
        """Envía ping cada 15 segundos para mantener la conexión activa."""
        while self.is_running:
            try:
                await asyncio.sleep(15)
                if self.ws and not self.ws.closed:
                    ping_msg = {"op": "ping", "ping": int(time.time())}
                    await self.ws.send(orjson.dumps(ping_msg).decode("utf-8"))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Error en ping heartbeat: {e}")
                break

    def stop(self):
        """Detiene el cliente WebSocket."""
        self.is_running = False
        if self.ws:
            asyncio.create_task(self.ws.close())

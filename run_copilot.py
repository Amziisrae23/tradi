import sys
import io

if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import os
import asyncio
import logging
from datetime import datetime, timezone
from aiohttp import web
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config.settings import settings
from src.exchanges.bitunix.client import BitunixClient
from src.patterns.smc_engine import SMCEngine
from src.simulation.monte_carlo import run_monte_carlo_simulation
from src.copilot.chart_renderer import ChartRenderer
from src.copilot.signal_generator import SignalGenerator
from src.copilot.telegram_notifier import TelegramNotifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TradiCopilot")
console = Console(force_terminal=True)

# Estado global para monitoreo HTTP
app_state = {
    "service": "Tradi Copilot 24/7",
    "status": "running",
    "last_scan": None,
    "signals_found": 0,
    "monitored_symbols": settings.DEFAULT_SYMBOLS
}

async def scan_market(client, smc, renderer, telegram):
    """Ejecuta una ronda de escaneo cuantitativo sobre Bitunix."""
    logger.info("Iniciando escaneo de mercado...")
    app_state["last_scan"] = datetime.now(timezone.utc).isoformat()

    all_setups = []
    for sym in settings.DEFAULT_SYMBOLS:
        try:
            df = client.get_historical_klines(sym, interval=settings.MTF_INTERVAL, limit=120)
            if df.empty or len(df) < 30:
                continue

            analysis = smc.analyze(sym, df)
            setups = analysis.get("setups", [])

            for s in setups:
                chart_path = renderer.render_trade_setup(df, s)
                mc = run_monte_carlo_simulation(win_rate=s.confidence_score / 100.0, reward_risk=s.rr_tp2)
                signal_text = SignalGenerator.format_signal_text(s, mc)
                all_setups.append((s, signal_text, chart_path))

                if telegram.is_configured():
                    logger.info(f"Enviando señal de {s.symbol} a Telegram...")
                    telegram.send_signal(signal_text, chart_path)

        except Exception as e:
            logger.error(f"Error escaneando {sym}: {e}")

    if all_setups:
        app_state["signals_found"] += len(all_setups)
        logger.info(f"¡{len(all_setups)} señales detectadas y despachadas!")
    else:
        logger.info("Escaneo completado: sin setups de alta confluencia en este ciclo.")

async def continuous_scanner_loop(client, smc, renderer, telegram, interval_seconds: int = 60):
    """Loop continuo que escanea el mercado cada 60 segundos."""
    logger.info("Iniciando loop continuo 24/7 de Tradi Copilot...")
    
    if telegram.is_configured():
        telegram.send_message("🟢 [TRADI COPILOT 24/7]: Sistema iniciado exitosamente en la nube. Monitoreando Bitunix en tiempo real...")

    while True:
        try:
            await scan_market(client, smc, renderer, telegram)
        except Exception as e:
            logger.error(f"Error en loop de escaneo: {e}")
        await asyncio.sleep(interval_seconds)

# Endpoints HTTP para Render / UptimeRobot
async def handle_health(request):
    return web.json_response(app_state)

async def init_app():
    app = web.Application()
    app.router.add_get("/", handle_health)
    app.router.add_get("/health", handle_health)
    return app

def main():
    client = BitunixClient()
    smc = SMCEngine(atr_period=settings.ATR_PERIOD)
    renderer = ChartRenderer()
    telegram = TelegramNotifier()

    port = int(os.getenv("PORT", "0"))

    if port > 0:
        logger.info(f"Iniciando en modo Cloud Web Service en puerto {port}...")
        loop = asyncio.get_event_loop()
        loop.create_task(continuous_scanner_loop(client, smc, renderer, telegram, interval_seconds=60))
        app = loop.run_until_complete(init_app())
        web.run_app(app, port=port, print=None)
    else:
        asyncio.run(scan_market(client, smc, renderer, telegram))

if __name__ == "__main__":
    main()

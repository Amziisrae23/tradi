import sys
import io

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import asyncio
import logging
from datetime import datetime, timezone
from aiohttp import web
from rich.console import Console

from config.settings import settings
from src.exchanges.bitunix.client import BitunixClient
from src.exchanges.bitunix.trader import trader
from src.patterns.smc_engine import SMCEngine
from src.simulation.monte_carlo import run_monte_carlo_simulation
from src.copilot.chart_renderer import ChartRenderer
from src.copilot.signal_generator import SignalGenerator
from src.copilot.telegram_notifier import TelegramNotifier
from src.copilot.position_monitor import position_monitor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TradiCopilot")
console = Console(force_terminal=True)

app_state = {
    "service": "Tradi Intelligent Copilot 24/7",
    "status": "running",
    "last_scan": None,
    "signals_found": 0,
    "executed_orders": 0,
    "monitored_symbols": settings.DEFAULT_SYMBOLS
}

async def process_telegram_actions(telegram: TelegramNotifier):
    """Revisa si el usuario presionó 'EJECUTAR' o 'DESCARTAR' en su Telegram."""
    clicks = telegram.check_button_clicks()
    for c in clicks:
        data = c.get("data", "")
        if data.startswith("exec:") or data.startswith("ex:"):
            parts = data.split(":")
            if len(parts) >= 5:
                sym = parts[1]
                direction = "LONG" if "L" in parts[2].upper() else "SHORT"
                try:
                    entry = float(parts[3])
                    sl = float(parts[4])
                    tp = float(parts[5]) if len(parts) >= 6 else (entry + 2.5 * abs(entry - sl))

                    logger.info(f"⚡ [TELEGRAM 1-CLIC]: Usuario confirmó ejecución de {sym} {direction}")
                    
                    # Ejecutar en Bitunix
                    res = trader.execute_order(
                        symbol=sym,
                        direction=direction,
                        entry_price=entry,
                        stop_loss=sl,
                        take_profit=tp,
                        risk_usd=settings.FIXED_RISK_USD
                    )
                    
                    app_state["executed_orders"] += 1
                    mode_label = "🟢 [ORDEN REAL EN BITUNIX]" if res["mode"] == "REAL" else "🧪 [ORDEN VIRTUAL SIMULADA]"
                    
                    receipt = f"""{mode_label}
⚡ 𝐏𝐀𝐑: {sym} | {direction}
📍 Entrada: ${entry:,.2f}
🛑 Stop Loss: ${sl:,.2f}
🎯 Take Profit: ${tp:,.2f}
💼 Riesgo asignado: ${settings.FIXED_RISK_USD:.2f} USD (2% Kelly)
🆔 ID: {res.get('order_id', 'N/A')}
ℹ️ {res.get('msg', 'Ejecutada con éxito')}"""
                    
                    telegram.send_message(receipt)

                    # Registrar la orden en el monitor de posiciones para alertas de Ganancia/Pérdida
                    tp1 = entry + (1.5 * abs(entry - sl)) if direction == "LONG" else entry - (1.5 * abs(entry - sl))
                    tp3 = entry + (5.0 * abs(entry - sl)) if direction == "LONG" else entry - (5.0 * abs(entry - sl))
                    position_monitor.register_trade(
                        order_id=res.get('order_id', 'N/A'),
                        symbol=sym,
                        direction=direction,
                        entry_price=entry,
                        stop_loss=sl,
                        tp1=tp1,
                        tp2=tp,
                        tp3=tp3,
                        risk_usd=settings.FIXED_RISK_USD
                    )

                except Exception as ex:
                    logger.error(f"Error procesando orden de Telegram: {ex}")
                    telegram.send_message(f"❌ Error al procesar la orden: {ex}")
        elif data == "discard":
            telegram.send_message("🗑️ Señal descartada.")

async def scan_market(client, smc, renderer, telegram):
    """Escanea las 10 criptomonedas más líquidas con Machine Learning y SMC."""
    logger.info(f"Iniciando escaneo inteligente de {len(settings.DEFAULT_SYMBOLS)} pares...")
    app_state["last_scan"] = datetime.now(timezone.utc).isoformat()

    all_setups = []
    latest_prices = {}

    for sym in settings.DEFAULT_SYMBOLS:
        try:
            df_mtf = client.get_historical_klines(sym, interval=settings.MTF_INTERVAL, limit=120)
            df_htf = client.get_historical_klines(sym, interval=settings.HTF_INTERVAL, limit=60)
            
            if df_mtf.empty or len(df_mtf) < 30:
                continue

            current_price = df_mtf['close'].iloc[-1]
            latest_prices[sym] = current_price

            analysis = smc.analyze(sym, df_mtf, df_htf)
            setups = analysis.get("setups", [])

            for s in setups:
                chart_path = renderer.render_trade_setup(df_mtf, s)
                mc = run_monte_carlo_simulation(win_rate=s.confidence_score / 100.0, reward_risk=s.rr_tp2)
                signal_text = SignalGenerator.format_signal_text(s, mc)
                all_setups.append((s, signal_text, chart_path))

                if telegram.is_configured():
                    logger.info(f"Enviando señal de {s.symbol} con botones de 1-Clic a Telegram...")
                    telegram.send_signal(signal_text, chart_path, setup=s)

        except Exception as e:
            logger.error(f"Error escaneando {sym}: {e}")

    # Monitorear posiciones activas y avisar si ganaron o perdieron
    if latest_prices:
        position_monitor.check_market_prices(latest_prices)

    if all_setups:
        app_state["signals_found"] += len(all_setups)
        logger.info(f"¡{len(all_setups)} señales de alta probabilidad detectadas y despachadas!")
    else:
        logger.info("Escaneo completado: mercado en balance, esperando confluencia institucional...")

async def continuous_scanner_loop(client, smc, renderer, telegram, interval_seconds: int = 60):
    logger.info("Iniciando loop continuo 24/7 con Machine Learning y 1-Clic Telegram...")
    
    if telegram.is_configured():
        telegram.send_message("🟢 [TRADI COPILOT 24/7]: Sistema inteligente activo. Escaneando los 10 pares más líquidos con Machine Learning y Monitoreo de Ganancias/Pérdidas.")

    while True:
        try:
            await process_telegram_actions(telegram)
            await scan_market(client, smc, renderer, telegram)
        except Exception as e:
            logger.error(f"Error en loop de escaneo: {e}")
        await asyncio.sleep(interval_seconds)

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

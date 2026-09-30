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
from src.intelligence.gemini_analyst import gemini_analyst

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TradiCopilot")
console = Console(force_terminal=True)

app_state = {
    "service": "Tradi Intelligent Copilot 24/7",
    "status": "running",
    "last_scan": None,
    "scans_completed": 0,
    "signals_found": 0,
    "executed_orders": 0,
    "monitored_symbols": settings.DEFAULT_SYMBOLS
}

async def process_telegram_actions(telegram: TelegramNotifier):
    """Revisa de forma asíncrona si el usuario presionó 'EJECUTAR' o 'DESCARTAR' en su Telegram."""
    clicks = await asyncio.to_thread(telegram.check_button_clicks)
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
                    
                    res = await asyncio.to_thread(
                        trader.execute_order,
                        symbol=sym,
                        direction=direction,
                        entry_price=entry,
                        stop_loss=sl,
                        take_profit=tp,
                        risk_usd=settings.FIXED_RISK_USD
                    )
                    
                    app_state["executed_orders"] += 1
                    mode_label = "🟢 [ORDEN REAL EN BITUNIX]" if res.get("mode") == "REAL" else "🧪 [ORDEN VIRTUAL SIMULADA]"
                    
                    receipt = f"""{mode_label}
⚡ 𝐏𝐀𝐑: {sym} | {direction}
📍 Entrada: ${entry:,.2f}
🛑 Stop Loss: ${sl:,.2f}
🎯 Take Profit: ${tp:,.2f}
💼 Riesgo asignado: ${settings.FIXED_RISK_USD:.2f} USD (2% Kelly)
🆔 ID: {res.get('order_id', 'N/A')}
ℹ️ {res.get('msg', 'Ejecutada con éxito')}"""
                    
                    await asyncio.to_thread(telegram.send_message, receipt)

                    tp1 = entry + (1.5 * abs(entry - sl)) if direction == "LONG" else entry - (1.5 * abs(entry - sl))
                    tp3 = entry + (5.0 * abs(entry - sl)) if direction == "LONG" else entry - (5.0 * abs(entry - sl))
                    
                    await asyncio.to_thread(
                        position_monitor.register_trade,
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
                    await asyncio.to_thread(telegram.send_message, f"❌ Error al procesar la orden: {ex}")
        elif data == "discard":
            await asyncio.to_thread(telegram.send_message, "🗑️ Señal descartada.")

async def scan_single_symbol(sym: str, client: BitunixClient, smc: SMCEngine, renderer: ChartRenderer, telegram: TelegramNotifier, already_signaled: set):
    """Escanea un único símbolo de forma asíncrona sin bloquear el event loop."""
    try:
        df_mtf = await asyncio.to_thread(client.get_historical_klines, sym, interval=settings.MTF_INTERVAL, limit=120)
        df_htf = await asyncio.to_thread(client.get_historical_klines, sym, interval=settings.HTF_INTERVAL, limit=60)

        if df_mtf.empty or len(df_mtf) < 30:
            return None, None

        current_price = float(df_mtf['close'].iloc[-1])
        analysis = await asyncio.to_thread(smc.analyze, sym, df_mtf, df_htf)
        setups = analysis.get("setups", [])

        dispatched = []
        for s in setups:
            # ── Deduplicación: un par solo emite UNA señal por ciclo de 60 min ──
            dedup_key = f"{sym}_{s.direction}"
            if dedup_key in already_signaled:
                logger.debug(f"Señal duplicada ignorada: {dedup_key}")
                continue
            already_signaled.add(dedup_key)

            chart_path = await asyncio.to_thread(renderer.render_trade_setup, df_mtf, s)
            mc = await asyncio.to_thread(run_monte_carlo_simulation, win_rate=s.confidence_score / 100.0, reward_risk=s.rr_tp2)
            signal_text = SignalGenerator.format_signal_text(s, mc)

            try:
                ai_analysis = await gemini_analyst.analyze_setup(s)
            except Exception as e:
                logger.warning(f"Error al analizar setup con Gemini para {s.symbol}: {e}")
                ai_analysis = None

            if ai_analysis and ai_analysis.strip():
                final_text = f"🤖 ANÁLISIS IA — {s.symbol}\n━━━━━━━━━━━━━━━━━━━━━\n\"{ai_analysis.strip()}\"\n\n{signal_text}"
            else:
                final_text = signal_text

            dispatched.append((s, final_text, chart_path))

            if telegram.is_configured():
                logger.info(f"Enviando señal de {s.symbol} con botones de 1-Clic a Telegram...")
                if hasattr(telegram, "send_signal"):
                    await asyncio.to_thread(telegram.send_signal, final_text, chart_path, setup=s)
                elif hasattr(telegram, "notify_signal"):
                    notify_fn = getattr(telegram, "notify_signal")
                    if asyncio.iscoroutinefunction(notify_fn):
                        await notify_fn(setup=s, chart_bytes=chart_path, signal_text=final_text)
                    else:
                        await asyncio.to_thread(notify_fn, setup=s, chart_bytes=chart_path, signal_text=final_text)

            # ── Registrar automáticamente en monitor para seguimiento TP/SL ──
            risk_usd = round(trader.account_equity * trader.default_risk_pct, 2)
            position_monitor.register_trade(
                order_id=f"SIG-{sym}-{s.direction[:1]}-{int(s.entry_price)}",
                symbol=sym,
                direction=s.direction,
                entry_price=s.entry_price,
                stop_loss=s.stop_loss,
                tp1=s.tp1,
                tp2=s.tp2,
                tp3=s.tp3,
                risk_usd=risk_usd
            )
            logger.info(f"📊 Trade registrado para monitoreo: {sym} {s.direction} | Riesgo: ${risk_usd:.2f}")

        return current_price, dispatched

    except Exception as e:
        logger.error(f"Error escaneando {sym}: {e}")
        return None, None


async def scan_market(client, smc, renderer, telegram):
    """Escanea las 10 criptomonedas más líquidas concurrentemente."""
    app_state["last_scan"] = datetime.now(timezone.utc).isoformat()
    app_state["scans_completed"] += 1

    all_setups = []
    latest_prices = {}
    already_signaled: set = set()  # Deduplicación por ciclo de escaneo

    for sym in settings.DEFAULT_SYMBOLS:
        price, setups = await scan_single_symbol(sym, client, smc, renderer, telegram, already_signaled)
        if price is not None:
            latest_prices[sym] = price
        if setups:
            all_setups.extend(setups)

    if latest_prices:
        await asyncio.to_thread(position_monitor.check_market_prices, latest_prices)

    if all_setups:
        app_state["signals_found"] += len(all_setups)
        logger.info(f"¡{len(all_setups)} señales de alta probabilidad detectadas y despachadas!")


async def continuous_scanner_loop(client, smc, renderer, telegram, interval_seconds: int = 60):
    """Loop continuo 24/7 garantizado sin pausas."""
    logger.info("Iniciando loop continuo 24/7 con Machine Learning y 1-Clic Telegram...")

    # Leer balance real de Bitunix al arrancar
    current_equity = await asyncio.to_thread(client.get_account_balance)
    risk_usd = round(current_equity * 0.02, 2)
    trader.update_account_equity(current_equity)
    logger.info(f"💰 Capital inicial leído de Bitunix: ${current_equity:,.2f} USDT | Riesgo/trade: ${risk_usd:.2f} USD")

    if telegram.is_configured():
        await asyncio.to_thread(
            telegram.send_message,
            f"🟢 [TRADI COPILOT 24/7]: Sistema inteligente activo en la nube.\n"
            f"💰 Capital detectado: ${current_equity:,.2f} USDT\n"
            f"⚖️ Riesgo por trade (2%): ${risk_usd:.2f} USD\n"
            f"📡 Escaneando los 10 pares de Bitunix en tiempo real cada 60 segundos."
        )

    loop_count = 0
    while True:
        try:
            await process_telegram_actions(telegram)
            await scan_market(client, smc, renderer, telegram)
            loop_count += 1

            # Cada 6 horas (360 ciclos de 60s): actualizar balance y enviar latido
            if loop_count % 360 == 0 and telegram.is_configured():
                current_equity = await asyncio.to_thread(client.get_account_balance)
                risk_usd = round(current_equity * 0.02, 2)
                trader.update_account_equity(current_equity)
                now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                await asyncio.to_thread(
                    telegram.send_message,
                    f"📡 [LATIDO TRADI — {now_str}]\n"
                    f"✅ Escáner 24/7 operativo · {app_state['scans_completed']} escaneos\n"
                    f"💰 Capital actualizado: ${current_equity:,.2f} USDT\n"
                    f"⚖️ Riesgo/trade activo: ${risk_usd:.2f} USD (2% dinámico)"
                )

        except Exception as e:
            logger.error(f"Error en loop de escaneo: {e}")

        await asyncio.sleep(interval_seconds)


async def handle_health(request):
    return web.json_response(app_state)

async def background_worker(app):
    """Gestor oficial de ciclo de vida para tareas en segundo plano en aiohttp."""
    client = BitunixClient()
    smc = SMCEngine(atr_period=settings.ATR_PERIOD)
    renderer = ChartRenderer()
    telegram = TelegramNotifier()

    app['scanner_task'] = asyncio.create_task(
        continuous_scanner_loop(client, smc, renderer, telegram, interval_seconds=60)
    )
    yield
    app['scanner_task'].cancel()
    try:
        await app['scanner_task']
    except asyncio.CancelledError:
        pass

def init_app():
    app = web.Application()
    app.router.add_get("/", handle_health)
    app.router.add_get("/health", handle_health)
    app.cleanup_ctx.append(background_worker)
    return app

def main():
    port = int(os.getenv("PORT", "0"))

    if port > 0:
        logger.info(f"Iniciando en modo Cloud Web Service en puerto {port}...")
        app = init_app()
        web.run_app(app, port=port, print=None)
    else:
        client = BitunixClient()
        smc = SMCEngine(atr_period=settings.ATR_PERIOD)
        renderer = ChartRenderer()
        telegram = TelegramNotifier()
        asyncio.run(scan_market(client, smc, renderer, telegram))

if __name__ == "__main__":
    main()

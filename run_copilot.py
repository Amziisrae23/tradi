import sys
import io

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import time
import asyncio
import logging
from typing import Dict, Any, Tuple, Optional, Set
from datetime import datetime, timezone
import pandas as pd
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
from src.intelligence.portfolio_risk import portfolio_risk
from src.storage.trading_ledger import ledger

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TradiCopilot")
console = Console(force_terminal=True)

app_state = {
    "service": "Tradi Intelligent Copilot 24/7",
    "status": "running",
    "execution_mode": settings.EXECUTION_MODE,
    "last_scan": None,
    "scans_completed": 0,
    "signals_found": 0,
    "executed_orders": 0,
    "monitored_symbols": settings.DEFAULT_SYMBOLS
}

async def process_telegram_actions(telegram: TelegramNotifier, client: BitunixClient):
    """Procesa de forma asíncrona tanto clics en botones como comandos de texto en Telegram."""
    updates = await asyncio.to_thread(telegram.check_all_updates)
    for u in updates:
        u_type = u.get("type")

        # ── 1. Comandos de texto (/balance, /stats, /mode, /pause, /resume, /historial, /help) ──
        if u_type == "command":
            cmd = u.get("text", "").strip().lower()
            user_name = u.get("user", "Usuario")

            if cmd == "/help" or cmd == "/ayuda" or cmd == "/start":
                help_msg = """🤖 𝗧𝗥𝗔𝗗𝗜 𝗖𝗢𝗣𝗜𝗟𝗢𝗧 | 𝗖𝗢𝗠𝗔𝗡𝗗𝗢𝗦 𝗗𝗜𝗦𝗣𝗢𝗡𝗜𝗕𝗟𝗘𝗦

💼 /balance ➔ Consulta saldo real en Bitunix y riesgo por trade
📊 /posiciones ➔ Muestra posiciones abiertas en Bitunix y su PnL en vivo
❌ /close <par> ➔ Cierra posición en Bitunix (ej. /close SOLUSDT o /close all)
📈 /stats ➔ Estadísticas contables (Win Rate, Profit Factor, P&L)
📜 /historial ➔ Últimos 5 trades registrados en el ledger
⚡ /mode auto ➔ Activa ejecución 100% autónoma (Manos Libres)
🕹️ /mode manual ➔ Activa modo copiloto (requiere clic de confirmación)
⏸️ /pause ➔ Pausa temporalmente nuevas aperturas de trades
▶️ /resume ➔ Reanuda el escáner y aperturas normales
ℹ️ /status ➔ Estado en vivo del bot y pares monitoreados"""
                await asyncio.to_thread(telegram.send_message, help_msg)

            elif cmd == "/balance" or cmd == "/saldo":
                bal = await asyncio.to_thread(client.get_account_balance)
                r_usd = round(bal * trader.default_risk_pct, 2)
                active_t = position_monitor.get_active_trades()
                bal_msg = f"""💰 𝗘𝗦𝗧𝗔𝗗𝗢 𝗗𝗘 𝗖𝗨𝗘𝗡𝗧𝗔 & 𝗖𝗔𝗣𝗜𝗧𝗔𝗟

💵 Capital Activo: ${bal:,.2f} USDT
🛡️ Riesgo por Trade (2% Kelly): ${r_usd:.2f} USD
⚡ Modo de Ejecución: {app_state['execution_mode']}
📈 Posiciones Monitoreadas ({len(active_t)}/{settings.MAX_CONCURRENT_TRADES}):"""
                if active_t:
                    for t in active_t:
                        bal_msg += f"\n  • {t['symbol']} {t['direction']} @ ${_fmt_price(t['entry_price'])} (SL: ${_fmt_price(t['current_sl'])})"
                else:
                    bal_msg += "\n  • Sin posiciones activas en este momento."
                await asyncio.to_thread(telegram.send_message, bal_msg)

            elif cmd == "/posiciones" or cmd == "/positions":
                open_pos = await asyncio.to_thread(trader.get_open_positions)
                if not open_pos:
                    await asyncio.to_thread(telegram.send_message, "📊 No hay posiciones abiertas actualmente en Bitunix.")
                else:
                    pos_msg = f"📊 𝗣𝗢𝗦𝗜𝗖𝗜𝗢𝗡𝗘𝗦 𝗔𝗕𝗜𝗘𝗥𝗧𝗔𝗦 𝗘𝗡 𝗕𝗜𝗧𝗨𝗡𝗜𝗫 ({len(open_pos)}):\n"
                    for p in open_pos:
                        sym = p.get('symbol')
                        side = p.get('side')
                        qty = p.get('qty')
                        margin = float(p.get('margin', 0))
                        entry_p = float(p.get('avgOpenPrice', 0))
                        pnl = float(p.get('unrealizedPNL', 0))
                        p_emoji = "🟢" if pnl >= 0 else "🔴"
                        pos_msg += f"\n• {sym} {side} ({qty} contratos)\n  📍 Entrada: ${_fmt_price(entry_p)} | Margen: ${margin:.2f} USDT\n  {p_emoji} PnL en vivo: ${pnl:+.2f} USD\n  🆔 ID: {p.get('positionId')}\n"
                    await asyncio.to_thread(telegram.send_message, pos_msg)

            elif cmd.startswith("/close") or cmd.startswith("/cerrar"):
                parts = cmd.split()
                if len(parts) >= 2:
                    target = parts[1].upper()
                    if target == "ALL" or target == "TODO":
                        results = await asyncio.to_thread(trader.close_all_positions)
                        await asyncio.to_thread(telegram.send_message, f"🚨 Todas las posiciones ({len(results)}) han sido cerradas en Bitunix.")
                    else:
                        open_pos = await asyncio.to_thread(trader.get_open_positions)
                        matched = [p for p in open_pos if target in p.get("symbol", "").upper()]
                        if matched:
                            for p in matched:
                                res = await asyncio.to_thread(trader.close_position, p["positionId"], p.get("symbol"))
                                await asyncio.to_thread(telegram.send_message, f"✔ {p.get('symbol')} cerrada en Bitunix: {res.get('msg')}")
                        else:
                            await asyncio.to_thread(telegram.send_message, f"⚠️ No se encontró posición abierta para {target} en Bitunix.")
                else:
                    await asyncio.to_thread(telegram.send_message, "ℹ️ Uso: /close <simbolo> (ej. /close LINKUSDT) o /close all")

            elif cmd == "/stats" or cmd == "/metricas":
                s = ledger.get_stats()
                stats_msg = f"""📊 𝗟𝗘𝗗𝗚𝗘𝗥 𝗖𝗨𝗔𝗡𝗧𝗜𝗧𝗔𝗧𝗜𝗩𝗢 — 𝗘𝗦𝗧𝗔𝗗Í𝗦𝗧𝗜𝗖𝗔𝗦

🎯 Win Rate: {s['win_rate']:.1f}% ({s['wins']}G - {s['losses']}P - {s['breakevens']}BE)
💵 P&L Total Neto: ${s['total_pnl_usd']:+.2f} USD ({s['total_pnl_r']:+.2f}R)
⚖️ Factor de Beneficio: {s['profit_factor']:.2f}
📈 Ganancia Promedio: +${s['avg_win_usd']:.2f} USD
📉 Pérdida Promedio: -${s['avg_loss_usd']:.2f} USD
🔢 Total Trades Auditados: {s['total_trades']}"""
                await asyncio.to_thread(telegram.send_message, stats_msg)

            elif cmd.startswith("/mode"):
                if "auto" in cmd:
                    app_state["execution_mode"] = "AUTO"
                    settings.EXECUTION_MODE = "AUTO"
                    await asyncio.to_thread(telegram.send_message, "⚡ [MODO AUTÓNOMO ACTIVADO]: El bot colocará órdenes en Bitunix automáticamente tras la aprobación de Gemini IA.")
                elif "manual" in cmd:
                    app_state["execution_mode"] = "MANUAL"
                    settings.EXECUTION_MODE = "MANUAL"
                    await asyncio.to_thread(telegram.send_message, "🕹️ [MODO COPILOTO ACTIVADO]: El bot enviará señales a Telegram con botones interactivos para confirmación previa.")

            elif cmd == "/pause" or cmd == "/pausar":
                portfolio_risk.is_paused = True
                await asyncio.to_thread(telegram.send_message, "⏸️ [BOT PAUSADO]: Se han congelado nuevas operaciones. Las posiciones abiertas continuarán siendo monitoreadas.")

            elif cmd == "/resume" or cmd == "/reanudar":
                portfolio_risk.is_paused = False
                await asyncio.to_thread(telegram.send_message, "▶️ [BOT REANUDADO]: El escáner y la apertura de operaciones están 100% operativos.")

            elif cmd == "/historial" or cmd == "/history":
                recent = ledger.get_recent_trades(limit=5)
                if not recent:
                    await asyncio.to_thread(telegram.send_message, "📜 No hay trades finalizados en el ledger todavía.")
                else:
                    hist_msg = "📜 𝗨́𝗟𝗧𝗜𝗠𝗢𝗦 𝗧𝗥𝗔𝗗𝗘𝗦 𝗔𝗨𝗗𝗜𝗧𝗔𝗗𝗢𝗦:\n"
                    for r in recent:
                        status_emoji = "✅" if r.get("pnl_usd", 0) > 0 else ("🛑" if r.get("pnl_usd", 0) < 0 else "⚪")
                        hist_msg += f"\n{status_emoji} {r['symbol']} {r['direction']} | PnL: ${r.get('pnl_usd', 0):+.2f} USD ({r['status']})"
                    await asyncio.to_thread(telegram.send_message, hist_msg)

            elif cmd == "/status" or cmd == "/estado":
                st_msg = f"""ℹ️ 𝗘𝗦𝗧𝗔𝗗𝗢 𝗗𝗘𝗟 𝗦𝗜𝗦𝗧𝗘𝗠𝗔 𝗧𝗥𝗔𝗗𝗜 𝟮𝟰/𝟳

🟢 Estado: {'PAUSADO ⏸️' if portfolio_risk.is_paused else 'OPERATIVO 🚀'}
⚙️ Modo: {app_state['execution_mode']}
📡 Escaneos Completados: {app_state['scans_completed']}
🎯 Señales Detectadas: {app_state['signals_found']}
💼 Órdenes Ejecutadas: {app_state['executed_orders']}
🛡️ Circuit Breaker Diario: Máx {settings.DAILY_LOSS_LIMIT_PCT*100:.0f}% pérdida"""
                await asyncio.to_thread(telegram.send_message, st_msg)

        # ── 2. Clics en Botones Interactivos (1-Clic Execution) ──
        elif u_type == "callback":
            data = u.get("data", "")
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

                        risk_usd = round(trader.account_equity * trader.default_risk_pct, 2)
                        res = await asyncio.to_thread(
                            trader.execute_order,
                            symbol=sym,
                            direction=direction,
                            entry_price=entry,
                            stop_loss=sl,
                            take_profit=tp,
                            risk_usd=risk_usd
                        )

                        if res.get("success"):
                            app_state["executed_orders"] += 1
                            mode_label = "🟢 [ORDEN REAL EN BITUNIX]" if res.get("mode") == "REAL" else "🧪 [ORDEN VIRTUAL SIMULADA]"

                            receipt = f"""{mode_label}
⚡ 𝐏𝐀𝐑: {sym} | {direction}
📍 Entrada: {_fmt_price(entry)}
🛑 Stop Loss: {_fmt_price(sl)}
🎯 Take Profit: {_fmt_price(tp)}
💼 Riesgo Máximo en SL: ${risk_usd:.2f} USD (Pérdida si toca SL)
🔒 Margen de Garantía Bitunix: ~${res.get('estimated_margin', 0.0):.2f} USDT (Colateral temporal)
📦 Valor de Posición: ${res.get('notional', 0.0):.2f} USD ({res.get('qty')} contratos)
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
                                risk_usd=risk_usd,
                                gemini_verdict="EJECUTAR",
                                gemini_reason="Aprobada por confirmación manual de 1-Clic"
                            )
                        else:
                            logger.warning(f"❌ [1-CLIC] Orden {sym} rechazada por Bitunix: {res.get('msg')}")
                            rejection_msg = f"""❌ 𝗢𝗥𝗗𝗘𝗡 𝗥𝗘𝗖𝗛𝗔𝗭𝗔𝗗𝗔 𝗣𝗢𝗥 𝗕𝗜𝗧𝗨𝗡𝗜𝗫
⚡ 𝐏𝐀𝐑: {sym} | {direction}
📍 Entrada: {_fmt_price(entry)}
🛑 Stop Loss: {_fmt_price(sl)}
⚠️ Motivo: {res.get('msg', 'Error desconocido')}
ℹ️ La orden NO fue ejecutada ni registrada en tus posiciones."""
                            await asyncio.to_thread(telegram.send_message, rejection_msg)

                    except Exception as ex:
                        logger.error(f"Error procesando orden de Telegram: {ex}")
                        await asyncio.to_thread(telegram.send_message, f"❌ Error al procesar la orden: {ex}")
            elif data == "discard":
                await asyncio.to_thread(telegram.send_message, "🗑️ Señal descartada.")


# ── Cooldown: máximo 1 señal por par cada 15 minutos (1 vela MTF) ────────────
_last_signal_time: Dict[str, float] = {}
SIGNAL_COOLDOWN_SECONDS = 900  # 15 minutos

def _check_macro_trend(symbol: str, df_htf: pd.DataFrame, direction: str) -> Tuple[bool, str]:
    """
    Pre-filtro de tendencia macro usando EMA50 en HTF (4H).
    Retorna (ok, razón).
    """
    if df_htf is None or df_htf.empty or len(df_htf) < 20:
        return True, "Sin datos HTF suficientes — se permite pasar"

    close = df_htf["close"].astype(float)
    # EMA adaptada a la longitud de velas disponibles (máx 50)
    span = min(50, len(close))
    ema50 = close.ewm(span=span, adjust=False).mean()
    current_price = close.iloc[-1]
    ema50_val = ema50.iloc[-1]

    # Pendiente de la EMA
    lookback = min(5, len(ema50) - 1)
    ema_slope = (ema50.iloc[-1] - ema50.iloc[-1 - lookback]) / ema50.iloc[-1 - lookback] * 100 if lookback > 0 else 0.0

    if direction == "LONG":
        if current_price < ema50_val * 0.995:  # Margen de 0.5%
            return False, f"{symbol} LONG RECHAZADO: precio ${_fmt_price(current_price)} BAJO EMA{span} 4H (${_fmt_price(ema50_val)}) — tendencia macro BAJISTA"
        if ema_slope < -0.5:
            return False, f"{symbol} LONG RECHAZADO: EMA{span} 4H cayendo {ema_slope:.2f}% — fuerte momentum bajista"
    else:  # SHORT
        if current_price > ema50_val * 1.005:
            return False, f"{symbol} SHORT RECHAZADO: precio ${_fmt_price(current_price)} SOBRE EMA{span} 4H (${_fmt_price(ema50_val)}) — tendencia macro ALCISTA"
        if ema_slope > 0.5:
            return False, f"{symbol} SHORT RECHAZADO: EMA{span} 4H subiendo {ema_slope:.2f}% — fuerte momentum alcista"

    return True, f"Tendencia macro alineada (precio vs EMA{span}: {((current_price/ema50_val)-1)*100:+.2f}%)"


def _fmt_price(val: Optional[float]) -> str:
    if val is None:
        return "N/A"
    if abs(val) < 0.001:
        return f"${val:.6f}"
    elif abs(val) < 1.0:
        return f"${val:.4f}"
    elif abs(val) < 10.0:
        return f"${val:.3f}"
    return f"${val:,.2f}"


async def scan_single_symbol(sym: str, client: BitunixClient, smc: SMCEngine, renderer: ChartRenderer, telegram: TelegramNotifier, already_signaled: set):
    """Escanea un único símbolo de forma asíncrona con triple filtro: SMC+ML → Macro EMA50 → Gemini Árbitro (Veto)."""
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
            # ── FILTRO 0: Deduplicación por ciclo de escaneo ──
            dedup_key = f"{sym}_{s.direction}"
            if dedup_key in already_signaled:
                logger.debug(f"Señal duplicada ignorada en el ciclo: {dedup_key}")
                continue
            already_signaled.add(dedup_key)

            # ── FILTRO 1: Cooldown temporal (1 señal cada 60 min por par) ──
            cooldown_key = f"{sym}_{s.direction}"
            last_t = _last_signal_time.get(cooldown_key, 0)
            elapsed = time.time() - last_t
            if elapsed < SIGNAL_COOLDOWN_SECONDS:
                remaining_min = int((SIGNAL_COOLDOWN_SECONDS - elapsed) / 60)
                logger.info(f"⏳ Cooldown activo para {cooldown_key}: faltan {remaining_min} min para nueva señal.")
                continue

            # ── FILTRO 2: Tendencia Macro HTF 4H con EMA50 ──
            macro_ok, macro_reason = _check_macro_trend(sym, df_htf, s.direction)
            if not macro_ok:
                logger.warning(f"🛑 [FILTRO MACRO 4H BLOQUEÓ]: {macro_reason}")
                continue
            logger.info(f"✅ [FILTRO MACRO 4H APROBÓ]: {macro_reason}")

            # ── FILTRO 3: Gemini 2.0 Flash como Árbitro con poder de VETO ──
            try:
                verdict, gem_reason, gem_analysis = await gemini_analyst.evaluate_setup(s)
            except Exception as e:
                logger.warning(f"Error consultando Gemini para {sym}: {e}")
                verdict, gem_reason, gem_analysis = "PASS", str(e), ""

            if verdict == "RECHAZAR":
                logger.warning(f"🤖🚫 [GEMINI VETÓ SEÑAL DE {sym} {s.direction}]: {gem_reason}")
                continue

            logger.info(f"🤖✅ [GEMINI APROBÓ SEÑAL DE {sym} {s.direction}]: {gem_reason}")

            # ── FILTRO 4: Gestor de Riesgo de Portafolio & Circuit Breaker ──
            active_trades_list = position_monitor.get_active_trades()
            risk_approved, risk_reason = portfolio_risk.can_open_trade(
                symbol=sym,
                direction=s.direction,
                active_trades=active_trades_list,
                account_equity=trader.account_equity
            )
            if not risk_approved:
                logger.warning(f"🛡️ [PORTFOLIO RISK BLOQUEÓ]: {risk_reason}")
                continue
            logger.info(f"🛡️ [PORTFOLIO RISK APROBÓ]: {risk_reason}")

            # ── Señal Aprobada: Renderizar Gráfico ──
            chart_path = await asyncio.to_thread(renderer.render_trade_setup, df_mtf, s)
            mc = await asyncio.to_thread(run_monte_carlo_simulation, win_rate=s.confidence_score / 100.0, reward_risk=s.rr_tp2)
            signal_text = SignalGenerator.format_signal_text(s, mc)

            if gem_analysis and gem_analysis.strip():
                final_text = f"🤖 ANÁLISIS IA (GEMINI) — {s.symbol}\n━━━━━━━━━━━━━━━━━━━━━\n\"{gem_analysis.strip()}\"\n\n{signal_text}"
            else:
                final_text = signal_text

            dispatched.append((s, final_text, chart_path))
            _last_signal_time[cooldown_key] = time.time()

            risk_usd = round(trader.account_equity * trader.default_risk_pct, 2)

            # ── MODO AUTÓNOMO vs MODO COPILOTO MANUAL ──
            if app_state["execution_mode"] == "AUTO":
                logger.info(f"⚡ [MODO AUTO]: Ejecutando orden de {s.symbol} {s.direction} automáticamente en Bitunix...")
                res = await asyncio.to_thread(
                    trader.execute_order,
                    symbol=s.symbol,
                    direction=s.direction,
                    entry_price=s.entry_price,
                    stop_loss=s.stop_loss,
                    take_profit=s.tp2,
                    risk_usd=risk_usd
                )
                if res.get("success"):
                    app_state["executed_orders"] += 1
                    mode_label = "🟢 [ORDEN REAL EN BITUNIX]" if res.get("mode") == "REAL" else "🧪 [ORDEN VIRTUAL SIMULADA]"

                    auto_receipt = f"""⚡ 𝗧𝗥𝗔𝗗𝗜 𝗔𝗨𝗧𝗢-𝗣𝗜𝗟𝗢𝗧 | 𝗢𝗥𝗗𝗘𝗡 𝗘𝗝𝗘𝗖𝗨𝗧𝗔𝗗𝗔 ⚡
{mode_label}
⚡ 𝐏𝐀𝐑: {s.symbol} | {s.direction}
📍 Entrada: ${_fmt_price(s.entry_price)}
🛑 Stop Loss: ${_fmt_price(s.stop_loss)}
🎯 Take Profit Principal (TP2): ${_fmt_price(s.tp2)}
💼 Riesgo Máximo en SL: ${risk_usd:.2f} USD (Pérdida si toca SL)
🔒 Margen de Garantía Bitunix: ~${res.get('estimated_margin', 0.0):.2f} USDT (Colateral temporal)
📦 Valor de Posición: ${res.get('notional', 0.0):.2f} USD ({res.get('qty')} contratos)
🤖 Veredicto Gemini: {gem_reason}
🆔 ID: {res.get('order_id', 'N/A')}

{final_text}"""
                    if telegram.is_configured():
                        if chart_path and os.path.exists(chart_path):
                            await asyncio.to_thread(telegram.send_photo, chart_path, caption=auto_receipt)
                        else:
                            await asyncio.to_thread(telegram.send_message, auto_receipt)

                    position_monitor.register_trade(
                        order_id=res.get('order_id', f"AUTO-{sym}-{s.direction[:1]}-{int(s.entry_price)}"),
                        symbol=sym,
                        direction=s.direction,
                        entry_price=s.entry_price,
                        stop_loss=s.stop_loss,
                        tp1=s.tp1,
                        tp2=s.tp2,
                        tp3=s.tp3,
                        risk_usd=risk_usd,
                        gemini_verdict=verdict,
                        gemini_reason=gem_reason
                    )
                    logger.info(f"📊 Trade registrado para monitoreo y ledger: {sym} {s.direction} | Riesgo: ${risk_usd:.2f}")
                else:
                    logger.warning(f"❌ [AUTO-PILOT] Orden {s.symbol} rechazada por Bitunix: {res.get('msg')}")
                    auto_reject_msg = f"""❌ 𝗧𝗥𝗔𝗗𝗜 𝗔𝗨𝗧𝗢-𝗣𝗜𝗟𝗢𝗧 | 𝗢𝗥𝗗𝗘𝗡 𝗥𝗘𝗖𝗛𝗔𝗭𝗔𝗗𝗔
⚡ 𝐏𝐀𝐑: {s.symbol} | {s.direction}
📍 Entrada: ${_fmt_price(s.entry_price)}
🛑 Stop Loss: ${_fmt_price(s.stop_loss)}
⚠️ Motivo del rechazo: {res.get('msg', 'Error desconocido')}
ℹ️ La orden no pudo colocarse en Bitunix y no fue registrada."""
                    if telegram.is_configured():
                        await asyncio.to_thread(telegram.send_message, auto_reject_msg)

            else:
                # Modo Manual: enviar señal con botones 1-Clic a Telegram
                if telegram.is_configured():
                    logger.info(f"Enviando señal APROBADA de {s.symbol} con botones de 1-Clic a Telegram...")
                    if hasattr(telegram, "send_signal"):
                        await asyncio.to_thread(telegram.send_signal, final_text, chart_path, setup=s)
                    elif hasattr(telegram, "notify_signal"):
                        notify_fn = getattr(telegram, "notify_signal")
                        if asyncio.iscoroutinefunction(notify_fn):
                            await notify_fn(setup=s, chart_bytes=chart_path, signal_text=final_text)
                        else:
                            await asyncio.to_thread(notify_fn, setup=s, chart_bytes=chart_path, signal_text=final_text)

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
            f"⚙️ Modo de Ejecución: {app_state['execution_mode']} (escribe /mode auto o /mode manual)\n"
            f"📡 Escaneando los 10 pares de Bitunix cada 60 segundos.\n"
            f"💬 Escribe /help en este chat para ver todos los comandos."
        )

    loop_count = 0
    while True:
        try:
            # 1. Reconciliar estado con posiciones reales en Bitunix
            await asyncio.to_thread(position_monitor.reconcile_with_exchange)

            # 2. Procesar botones interactivos y comandos de texto en Telegram
            await process_telegram_actions(telegram, client)

            # 3. Escaneo de mercado y monitoreo de precios en vivo
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

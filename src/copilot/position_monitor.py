import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from config.settings import settings
from src.intelligence.ml_model import brain
from src.copilot.telegram_notifier import TelegramNotifier
from src.storage.trading_ledger import ledger

logger = logging.getLogger("PositionMonitor")
ACTIVE_TRADES_FILE = os.path.join("data", "active_trades.json")

class PositionMonitor:
    """Monitorea el estado de las operaciones abiertas y envía notificaciones de Ganancia/Pérdida a Telegram."""

    def __init__(self, telegram: Optional[TelegramNotifier] = None):
        self.telegram = telegram or TelegramNotifier()
        self.active_trades = self._load_active_trades()

    def get_active_trades(self) -> List[Dict[str, Any]]:
        """Retorna las operaciones actualmente abiertas."""
        return [t for t in self.active_trades if t.get("status") == "OPEN"]

    def _load_active_trades(self) -> List[Dict[str, Any]]:
        if os.path.exists(ACTIVE_TRADES_FILE):
            try:
                with open(ACTIVE_TRADES_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error cargando trades activos: {e}")
        return []

    def _save_active_trades(self):
        try:
            with open(ACTIVE_TRADES_FILE, "w", encoding="utf-8") as f:
                json.dump(self.active_trades, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando trades activos: {e}")

    def register_trade(
        self,
        order_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        tp1: float,
        tp2: float,
        tp3: float,
        risk_usd: float = 10.0,
        gemini_verdict: str = "EJECUTAR",
        gemini_reason: str = ""
    ):
        """Registra una nueva orden abierta para su seguimiento continuo y auditoría contable."""
        trade = {
            "order_id": order_id,
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "current_sl": stop_loss,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "risk_usd": risk_usd,
            "tp1_hit": False,
            "tp2_hit": False,
            "status": "OPEN",
            "opened_at": datetime.now(timezone.utc).isoformat()
        }
        self.active_trades.append(trade)
        self._save_active_trades()

        # Auditoría contable en SQLite
        ledger.record_trade_opened(
            order_id=order_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            risk_usd=risk_usd,
            gemini_verdict=gemini_verdict,
            gemini_reason=gemini_reason
        )
        logger.info(f"Trade registrado para monitoreo y ledger: {symbol} {direction} (ID: {order_id})")

    def check_market_prices(self, current_prices: Dict[str, float]):
        """Comprueba si el precio actual tocó Take Profit o Stop Loss y notifica en Telegram."""
        remaining_trades = []

        for trade in self.active_trades:
            sym = trade["symbol"]
            price = current_prices.get(sym)
            if not price:
                remaining_trades.append(trade)
                continue

            direction = trade["direction"]
            entry = trade["entry_price"]
            sl = trade["current_sl"]
            tp1 = trade["tp1"]
            tp2 = trade["tp2"]
            tp3 = trade["tp3"]
            risk_usd = trade["risk_usd"]

            closed = False

            # === CHEQUEO LONG ===
            if direction == "LONG":
                # 1. Checar Stop Loss
                if price <= sl:
                    closed = True
                    if trade["tp1_hit"]:
                        pnl_u = round(risk_usd * 0.6, 2)
                        msg = f"🛡️ [CIERRE EN BREAKEVEN]: {sym} tocó tu punto de entrada protegido. Ganancia neta asegurada: +${pnl_u:.2f} USD."
                        brain.update_feedback(sym, "WIN", actual_rr=0.6)
                        ledger.record_trade_closed(order_id=trade["order_id"], exit_price=entry, pnl_usd=pnl_u, pnl_r=0.6, status="BREAKEVEN")
                    else:
                        pnl_u = -round(risk_usd, 2)
                        msg = f"🛑 [STOP LOSS EJECUTADO]: {sym} tocó el SL en ${sl:,.2f}.\n💼 Pérdida controlada: -${risk_usd:.2f} USD."
                        brain.update_feedback(sym, "LOSS", actual_rr=-1.0)
                        ledger.record_trade_closed(order_id=trade["order_id"], exit_price=sl, pnl_usd=pnl_u, pnl_r=-1.0, status="STOP_LOSS")
                    self.telegram.send_message(msg)

                # 2. Checar TP1 (1.5R)
                elif not trade["tp1_hit"] and price >= tp1:
                    trade["tp1_hit"] = True
                    trade["current_sl"] = entry  # Mover SL a Breakeven
                    msg = f"🎉 [TP 1 ALCANZADO (+1.5R)]: {sym} superó ${tp1:,.2f}!\n💰 Ganancia del 40% asegurada: +${risk_usd * 0.6:.2f} USD.\n🛡️ Stop Loss movido a Breakeven (${entry:,.2f}) - ¡Trade libre de riesgo!"
                    self.telegram.send_message(msg)

                # 3. Checar TP2 (3.0R - Target Principal)
                elif trade["tp1_hit"] and not trade["tp2_hit"] and price >= tp2:
                    trade["tp2_hit"] = True
                    msg = f"🎯 [OBJETIVO PRINCIPAL TP 2 (+3.0R)]: {sym} alcanzó ${tp2:,.2f}!\n💰 Ganancia acumulada: +${risk_usd * 1.8:.2f} USD."
                    self.telegram.send_message(msg)

                # 4. Checar TP3 (5.0R - Runner Final)
                elif trade["tp2_hit"] and price >= tp3:
                    closed = True
                    pnl_u = round(risk_usd * 2.8, 2)
                    msg = f"🚀 [TP 3 RUNNER MÁXIMO (+5.0R)]: {sym} alcanzó ${tp3:,.2f}!\n🏆 Trade 100% completado con éxito: +${pnl_u:.2f} USD netos."
                    brain.update_feedback(sym, "WIN", actual_rr=2.8)
                    ledger.record_trade_closed(order_id=trade["order_id"], exit_price=tp3, pnl_usd=pnl_u, pnl_r=2.8, status="TP3_TARGET")
                    self.telegram.send_message(msg)

            # === CHEQUEO SHORT ===
            elif direction == "SHORT":
                if price >= sl:
                    closed = True
                    if trade["tp1_hit"]:
                        pnl_u = round(risk_usd * 0.6, 2)
                        msg = f"🛡️ [CIERRE EN BREAKEVEN]: {sym} tocó tu punto de entrada protegido. Ganancia neta asegurada: +${pnl_u:.2f} USD."
                        brain.update_feedback(sym, "WIN", actual_rr=0.6)
                        ledger.record_trade_closed(order_id=trade["order_id"], exit_price=entry, pnl_usd=pnl_u, pnl_r=0.6, status="BREAKEVEN")
                    else:
                        pnl_u = -round(risk_usd, 2)
                        msg = f"🛑 [STOP LOSS EJECUTADO]: {sym} tocó el SL en ${sl:,.2f}.\n💼 Pérdida controlada: -${risk_usd:.2f} USD."
                        brain.update_feedback(sym, "LOSS", actual_rr=-1.0)
                        ledger.record_trade_closed(order_id=trade["order_id"], exit_price=sl, pnl_usd=pnl_u, pnl_r=-1.0, status="STOP_LOSS")
                    self.telegram.send_message(msg)

                elif not trade["tp1_hit"] and price <= tp1:
                    trade["tp1_hit"] = True
                    trade["current_sl"] = entry
                    msg = f"🎉 [TP 1 ALCANZADO (+1.5R)]: {sym} cayó a ${tp1:,.2f}!\n💰 Ganancia del 40% asegurada: +${risk_usd * 0.6:.2f} USD.\n🛡️ Stop Loss movido a Breakeven (${entry:,.2f}) - ¡Trade libre de riesgo!"
                    self.telegram.send_message(msg)

                elif trade["tp1_hit"] and not trade["tp2_hit"] and price <= tp2:
                    trade["tp2_hit"] = True
                    msg = f"🎯 [OBJETIVO PRINCIPAL TP 2 (+3.0R)]: {sym} cayó a ${tp2:,.2f}!\n💰 Ganancia acumulada: +${risk_usd * 1.8:.2f} USD."
                    self.telegram.send_message(msg)

                elif trade["tp2_hit"] and price <= tp3:
                    closed = True
                    pnl_u = round(risk_usd * 2.8, 2)
                    msg = f"🚀 [TP 3 RUNNER MÁXIMO (+5.0R)]: {sym} cayó a ${tp3:,.2f}!\n🏆 Trade 100% completado con éxito: +${pnl_u:.2f} USD netos."
                    brain.update_feedback(sym, "WIN", actual_rr=2.8)
                    ledger.record_trade_closed(order_id=trade["order_id"], exit_price=tp3, pnl_usd=pnl_u, pnl_r=2.8, status="TP3_TARGET")
                    self.telegram.send_message(msg)

            if not closed:
                remaining_trades.append(trade)

        self.active_trades = remaining_trades
        self._save_active_trades()

position_monitor = PositionMonitor()

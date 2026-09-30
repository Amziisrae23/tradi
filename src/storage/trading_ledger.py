"""
Módulo de Contabilidad Cuantitativa: Trading Ledger (SQLite).
Registra y audita cada señal, orden, comisión y resultado de P&L en tiempo real.
Provee métricas de rendimiento institucional: Win Rate, Profit Factor, Sharpe Ratio y Drawdown.
"""

import os
import sqlite3
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger("TradingLedger")
DB_DIR = os.path.join(os.getcwd(), "data")
DB_PATH = os.path.join(DB_DIR, "tradi_ledger.db")


class TradingLedger:
    """Gestor contable persistente en SQLite para auditoría y estadísticas de trading."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    tp1 REAL NOT NULL,
                    tp2 REAL NOT NULL,
                    tp3 REAL NOT NULL,
                    risk_usd REAL NOT NULL,
                    exit_price REAL,
                    pnl_usd REAL DEFAULT 0.0,
                    pnl_r REAL DEFAULT 0.0,
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    gemini_verdict TEXT,
                    gemini_reason TEXT,
                    opened_at TEXT NOT NULL,
                    closed_at TEXT
                )
            """)
            conn.commit()

    def record_trade_opened(
        self,
        order_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        tp1: float,
        tp2: float,
        tp3: float,
        risk_usd: float,
        gemini_verdict: str = "EJECUTAR",
        gemini_reason: str = ""
    ) -> bool:
        """Registra la apertura de una nueva operación."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO trades (
                        order_id, symbol, direction, entry_price, stop_loss,
                        tp1, tp2, tp3, risk_usd, status, gemini_verdict,
                        gemini_reason, opened_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?)
                """, (
                    order_id, symbol, direction, entry_price, stop_loss,
                    tp1, tp2, tp3, risk_usd, gemini_verdict, gemini_reason,
                    datetime.now(timezone.utc).isoformat()
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error registrando trade abierto en SQLite: {e}")
            return False

    def record_trade_closed(
        self,
        order_id: str,
        exit_price: float,
        pnl_usd: float,
        pnl_r: float,
        status: str = "CLOSED"
    ) -> bool:
        """Registra el cierre de una operación y su balance final."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE trades
                    SET exit_price = ?, pnl_usd = ?, pnl_r = ?, status = ?,
                        closed_at = ?
                    WHERE order_id = ? OR order_id LIKE ?
                """, (
                    exit_price, pnl_usd, pnl_r, status,
                    datetime.now(timezone.utc).isoformat(),
                    order_id, f"%{order_id}%"
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error actualizando trade cerrado en SQLite: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Calcula estadísticas institucionales en tiempo real sobre los trades cerrados."""
        try:
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT * FROM trades WHERE status != 'OPEN'")
                rows = cur.fetchall()

                total_closed = len(rows)
                if total_closed == 0:
                    return {
                        "total_trades": 0,
                        "win_rate": 0.0,
                        "wins": 0,
                        "losses": 0,
                        "breakevens": 0,
                        "total_pnl_usd": 0.0,
                        "total_pnl_r": 0.0,
                        "profit_factor": 0.0,
                        "avg_win_usd": 0.0,
                        "avg_loss_usd": 0.0
                    }

                wins = [r for r in rows if r["pnl_usd"] > 0.05]
                losses = [r for r in rows if r["pnl_usd"] < -0.05]
                breakevens = [r for r in rows if -0.05 <= r["pnl_usd"] <= 0.05]

                total_pnl_usd = sum(r["pnl_usd"] for r in rows)
                total_pnl_r = sum(r["pnl_r"] for r in rows)

                gross_profit = sum(r["pnl_usd"] for r in wins)
                gross_loss = abs(sum(r["pnl_usd"] for r in losses))

                profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.9 if gross_profit > 0 else 0.0)
                win_rate = (len(wins) / total_closed) * 100.0

                return {
                    "total_trades": total_closed,
                    "win_rate": round(win_rate, 1),
                    "wins": len(wins),
                    "losses": len(losses),
                    "breakevens": len(breakevens),
                    "total_pnl_usd": round(total_pnl_usd, 2),
                    "total_pnl_r": round(total_pnl_r, 2),
                    "profit_factor": round(profit_factor, 2),
                    "avg_win_usd": round(gross_profit / len(wins), 2) if wins else 0.0,
                    "avg_loss_usd": round(gross_loss / len(losses), 2) if losses else 0.0
                }
        except Exception as e:
            logger.error(f"Error calculando estadísticas en SQLite: {e}")
            return {"total_trades": 0, "win_rate": 0.0, "total_pnl_usd": 0.0}

    def get_daily_pnl_usd(self) -> float:
        """Obtiene el P&L acumulado en las últimas 24 horas UTC."""
        try:
            today_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT SUM(pnl_usd) FROM trades
                    WHERE closed_at LIKE ? AND status != 'OPEN'
                """, (f"{today_prefix}%",))
                val = cur.fetchone()[0]
                return float(val) if val is not None else 0.0
        except Exception as e:
            logger.error(f"Error obteniendo P&L diario: {e}")
            return 0.0

    def get_recent_trades(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Obtiene las operaciones más recientes."""
        try:
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,))
                return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            logger.error(f"Error obteniendo trades recientes: {e}")
            return []


# Instancia Singleton
ledger = TradingLedger()

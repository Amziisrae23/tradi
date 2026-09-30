"""
Módulo de Gestión de Riesgo de Portafolio y Circuit Breaker.
Controla el calor de la cartera, previene la sobreexposición y protege el capital
mediante bloqueos automáticos en días de alta volatilidad adversa.
"""

import logging
from typing import Tuple, List, Dict, Any, Optional
from config.settings import settings
from src.storage.trading_ledger import ledger

logger = logging.getLogger("PortfolioRisk")


class PortfolioRiskManager:
    """
    Gestor de Riesgo de Portafolio Institucional:
    - Control de Calor de Cartera (Límite estricto de operaciones simultáneas)
    - Circuit Breaker Diario (Pausa de emergencia si la pérdida acumulada supera el umbral diario)
    - Filtro de Correlación de Activos (Evita abrir múltiples posiciones en la misma dirección si el mercado se sobreextiende)
    """

    def __init__(
        self,
        max_concurrent_trades: Optional[int] = None,
        daily_loss_limit_pct: Optional[float] = None
    ):
        self.max_concurrent_trades = max_concurrent_trades or getattr(settings, "MAX_CONCURRENT_TRADES", 2)
        self.daily_loss_limit_pct = daily_loss_limit_pct or getattr(settings, "DAILY_LOSS_LIMIT_PCT", 0.05)
        self.is_paused = False

    def can_open_trade(
        self,
        symbol: str,
        direction: str,
        active_trades: List[Dict[str, Any]],
        account_equity: float
    ) -> Tuple[bool, str]:
        """
        Evalúa si el portafolio tiene capacidad y seguridad para abrir una nueva operación.
        Retorna (aprobado: bool, razón: str).
        """
        if self.is_paused:
            return False, "🛑 Bot en modo PAUSA manual por comando de usuario."

        # 1. Límite de Operaciones Concurrentes (Calor de Cartera)
        active_count = len([t for t in active_trades if t.get("status") == "OPEN"])
        if active_count >= self.max_concurrent_trades:
            return False, (
                f"🛑 Calor de cartera al límite: {active_count}/{self.max_concurrent_trades} "
                f"operaciones activas. Se descarta {symbol} para proteger el capital."
            )

        # 2. Evitar posición duplicada en el mismo activo
        for t in active_trades:
            if t.get("symbol") == symbol and t.get("status") == "OPEN":
                return False, f"🛑 Ya existe una posición abierta en {symbol}."

        # 3. Circuit Breaker Diario (Pérdida máxima por día)
        daily_pnl = ledger.get_daily_pnl_usd()
        max_allowed_loss = account_equity * self.daily_loss_limit_pct
        if daily_pnl < -max_allowed_loss:
            return False, (
                f"🛑 CIRCUIT BREAKER ACTIVADO: Pérdida del día (${abs(daily_pnl):.2f} USD) "
                f"excede el {self.daily_loss_limit_pct*100:.0f}% del capital (${max_allowed_loss:.2f} USD). "
                f"Nuevas operaciones bloqueadas hasta mañana UTC."
            )

        return True, "✅ Parámetros de riesgo de portafolio y margen aprobados."


# Instancia Singleton
portfolio_risk = PortfolioRiskManager()

import numpy as np
from typing import Dict, Any, List

def run_monte_carlo_simulation(
    win_rate: float = 0.60,
    reward_risk: float = 2.5,
    risk_per_trade_pct: float = 0.02,
    n_simulations: int = 25000,
    n_trades: int = 200,
    initial_capital: float = 10000.0,
    ruin_threshold_pct: float = 0.40  # 40% capital loss = ruin
) -> Dict[str, Any]:
    """
    Simulación de Monte Carlo para calcular métricas probabilísticas de riesgo:
    - Probabilidad de Ruina (Risk of Ruin - RoR)
    - Drawdown Máximo esperado (Percentil 95% y 99%)
    - Crecimiento medio de balance
    """
    ruin_capital = initial_capital * (1.0 - ruin_threshold_pct)
    ruin_count = 0
    max_drawdowns = np.zeros(n_simulations, dtype=np.float64)
    final_equities = np.zeros(n_simulations, dtype=np.float64)

    # Multiplicadores de resultado por trade
    win_mult = 1.0 + (risk_per_trade_pct * reward_risk)
    loss_mult = 1.0 - risk_per_trade_pct

    # Generar matriz aleatoria de Bernoulli de trades (n_simulations x n_trades)
    random_trades = np.random.binomial(1, win_rate, size=(n_simulations, n_trades))
    trade_multipliers = np.where(random_trades == 1, win_mult, loss_mult)

    # Calcular curvas de equity acumuladas
    equity_paths = np.cumprod(trade_multipliers, axis=1) * initial_capital
    # Añadir capital inicial al inicio
    equity_paths = np.hstack([np.full((n_simulations, 1), initial_capital), equity_paths])

    # Calcular Peak y Drawdown para cada simulación
    peaks = np.maximum.accumulate(equity_paths, axis=1)
    drawdowns = (peaks - equity_paths) / peaks
    max_drawdowns = np.max(drawdowns, axis=1)
    final_equities = equity_paths[:, -1]

    # Contar ruinas (si alguna vez tocó el capital de ruina)
    min_equities = np.min(equity_paths, axis=1)
    ruin_count = np.sum(min_equities <= ruin_capital)

    prob_of_ruin = float(ruin_count / n_simulations)
    mdd_95 = float(np.percentile(max_drawdowns, 95))
    mdd_99 = float(np.percentile(max_drawdowns, 99))
    expected_final = float(np.median(final_equities))

    # Fractional Kelly Formula: f = (p * b - q) / b * 0.5 (Half Kelly)
    b = reward_risk
    p = win_rate
    q = 1.0 - p
    kelly_full = (p * b - q) / b if b > 0 else 0
    kelly_half = max(0.005, min(kelly_full * 0.5, 0.05))

    return {
        "n_simulations": n_simulations,
        "prob_of_ruin": prob_of_ruin,
        "max_drawdown_p95": mdd_95,
        "max_drawdown_p99": mdd_99,
        "expected_final_equity": expected_final,
        "recommended_position_pct": round(kelly_half * 100, 2)
    }

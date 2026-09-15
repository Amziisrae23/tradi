import numpy as np
from typing import Dict, Any, List, Optional

def run_monte_carlo_simulation(
    win_rate: float = 0.58,
    reward_risk: float = 2.6,
    risk_per_trade_pct: float = 0.02,
    fee_per_trade_pct: float = 0.0008,
    n_simulations: int = 100000,
    n_trades: int = 200,
    initial_capital: float = 500.0,
    ruin_threshold_pct: float = 0.50,  # 50% drawdown ($250) = ruina
    empirical_returns: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Simulación de Monte Carlo Bootstrap (100,000 Caminos) para calcular:
    - Probabilidad de Ruina (Risk of Ruin - RoR)
    - Drawdown Máximo esperado (Percentil 95% y 99%)
    - Distribución percentil de capital final (P5, P10, P25, Mediana, P75, P90, P95)
    - Rebalanceo óptimo mediante Criterio de Half-Kelly
    """
    ruin_capital = initial_capital * (1.0 - ruin_threshold_pct)

    if empirical_returns is not None and len(empirical_returns) >= 3:
        # Bootstrap resampling a partir de los retornos empíricos del backtest
        ret_array = np.array(empirical_returns)
        # Asumiendo un riesgo base en dólares de $10 USD (2% de $500), derivar los R-múltiplos empíricos
        base_risk_usd = initial_capital * 0.02
        r_multiples = ret_array / base_risk_usd
        # Muestreo con reemplazo para 100,000 caminos x n_trades
        resampled_indices = np.random.randint(0, len(r_multiples), size=(n_simulations, n_trades))
        # Escalar los retornos empíricos al porcentaje de riesgo exacto de cada perfil (1%, 2%, 5%)
        trade_returns = r_multiples[resampled_indices] * risk_per_trade_pct
        trade_multipliers = 1.0 + trade_returns
    else:
        # Modelo Paramétrico Bernoulli con deducción de comisiones
        win_mult = 1.0 + (risk_per_trade_pct * reward_risk) - fee_per_trade_pct
        loss_mult = 1.0 - risk_per_trade_pct - fee_per_trade_pct
        random_trades = np.random.binomial(1, win_rate, size=(n_simulations, n_trades))
        trade_multipliers = np.where(random_trades == 1, win_mult, loss_mult)

    # Calcular curvas acumuladas de balance
    equity_paths = np.cumprod(trade_multipliers, axis=1) * initial_capital
    equity_paths = np.hstack([np.full((n_simulations, 1), initial_capital), equity_paths])

    # Calcular Drawdowns
    peaks = np.maximum.accumulate(equity_paths, axis=1)
    drawdowns = (peaks - equity_paths) / peaks
    max_drawdowns = np.max(drawdowns, axis=1)
    final_equities = equity_paths[:, -1]

    # Contar eventos de ruina (< $250)
    min_equities = np.min(equity_paths, axis=1)
    ruin_count = np.sum(min_equities <= ruin_capital)

    prob_of_ruin = float(ruin_count / n_simulations)
    mdd_median = float(np.median(max_drawdowns))
    mdd_95 = float(np.percentile(max_drawdowns, 95))
    mdd_99 = float(np.percentile(max_drawdowns, 99))
    
    p5_final = float(np.percentile(final_equities, 5))
    p10_final = float(np.percentile(final_equities, 10))
    p25_final = float(np.percentile(final_equities, 25))
    expected_final = float(np.median(final_equities))
    p75_final = float(np.percentile(final_equities, 75))
    p90_final = float(np.percentile(final_equities, 90))
    p95_final = float(np.percentile(final_equities, 95))

    # Criterio de Half-Kelly: f* = (p * b - q) / b * 0.5
    b = reward_risk
    p = win_rate
    q = 1.0 - p
    kelly_full = (p * b - q) / b if b > 0 else 0.0
    kelly_half = max(0.005, min(kelly_full * 0.5, 0.05))

    res = {
        "n_simulations": n_simulations,
        "n_trades": n_trades,
        "prob_of_ruin": prob_of_ruin,
        "max_drawdown_median": mdd_median,
        "max_drawdown_p95": mdd_95,
        "max_drawdown_p99": mdd_99,
        "p5_final_equity": p5_final,
        "p10_final_equity": p10_final,
        "p25_final_equity": p25_final,
        "expected_final_equity": expected_final,
        "p75_final_equity": p75_final,
        "p90_final_equity": p90_final,
        "p95_final_equity": p95_final,
        "recommended_position_pct": round(kelly_half * 100, 2),
        "equity_paths_sample": equity_paths[:500],
        # Backward compatibility aliases
        "median_final": expected_final,
        "p10_final": p10_final,
        "p25_final": p25_final,
        "p75_final": p75_final,
        "p90_final": p90_final,
        "max_dd_p95": mdd_95,
        "max_dd_median": mdd_median
    }
    return res


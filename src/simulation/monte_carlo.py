import numpy as np
from typing import Dict, Any, List, Optional

def calculate_time_to_milestones(
    equity_paths: np.ndarray,
    milestones: Optional[List[float]] = None,
    trades_per_week: float = 6.0
) -> Dict[str, Dict[str, Any]]:
    """
    Calcula la estimación temporal probabilística (P10, P50 Mediana, P90)
    en número de operaciones, semanas y meses para alcanzar los hitos clave de capital
    ($1,000, $2,500, $5,000, $10,000, $25,000 USD) a partir del capital inicial.
    
    Parámetros:
        equity_paths: Matriz numpy de dimensión (n_simulations, n_trades + 1)
        milestones: Lista de objetivos de capital en USD
        trades_per_week: Frecuencia empírica promedio de señales por semana (default: 6.0)
    """
    if milestones is None:
        milestones = [1000.0, 2500.0, 5000.0, 10000.0, 25000.0]

    n_sims, n_steps = equity_paths.shape
    milestone_results = {}

    for m in milestones:
        reached = equity_paths >= m
        any_reached = np.any(reached, axis=1)
        prob_reaching = float(np.sum(any_reached) / n_sims)

        first_indices = np.argmax(reached, axis=1)
        valid_trades = first_indices[any_reached]

        if len(valid_trades) > 0:
            p10_trades = int(np.percentile(valid_trades, 10))
            p50_trades = int(np.percentile(valid_trades, 50))
            p90_trades = int(np.percentile(valid_trades, 90))

            p10_weeks = round(p10_trades / trades_per_week, 1)
            p50_weeks = round(p50_trades / trades_per_week, 1)
            p90_weeks = round(p90_trades / trades_per_week, 1)

            p10_months = round(p10_weeks / 4.33, 1)
            p50_months = round(p50_weeks / 4.33, 1)
            p90_months = round(p90_weeks / 4.33, 1)
        else:
            p10_trades = p50_trades = p90_trades = None
            p10_weeks = p50_weeks = p90_weeks = None
            p10_months = p50_months = p90_months = None

        milestone_results[f"${int(m):,} USD"] = {
            "target_usd": m,
            "prob_reached": prob_reaching,
            "p10_trades": p10_trades,
            "p50_trades_median": p50_trades,
            "p90_trades": p90_trades,
            "p10_weeks": p10_weeks,
            "p50_weeks_median": p50_weeks,
            "p90_weeks": p90_weeks,
            "p10_months": p10_months,
            "p50_months_median": p50_months,
            "p90_months": p90_months
        }

    return milestone_results

def run_monte_carlo_simulation(
    win_rate: float = 0.58,
    reward_risk: float = 2.6,
    risk_per_trade_pct: float = 0.02,
    fee_per_trade_pct: float = 0.0008,
    n_simulations: int = 100000,
    n_trades: int = 200,
    initial_capital: float = 500.0,
    ruin_threshold_pct: float = 0.50,  # 50% drawdown ($250) = ruina
    trades_per_week: float = 6.0,
    milestones: Optional[List[float]] = None,
    empirical_returns: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Simulación de Monte Carlo Bootstrap (100,000 Caminos) con Gestión de Crecimiento Compuesto Dinámico:
    - Probabilidad de Ruina (Risk of Ruin - RoR)
    - Drawdown Máximo esperado (Percentil 95% y 99%)
    - Distribución percentil de capital final (P5, P10, P25, Mediana, P75, P90, P95)
    - Rebalanceo óptimo mediante Criterio de Half-Kelly
    - Estimación de tiempo a hitos ($1k, $2.5k, $5k, $10k, $25k) en semanas y meses
    """
    ruin_capital = initial_capital * (1.0 - ruin_threshold_pct)

    if empirical_returns is not None and len(empirical_returns) >= 3:
        # Bootstrap resampling a partir de los retornos empíricos del backtest
        ret_array = np.array(empirical_returns)
        base_risk_usd = initial_capital * 0.02
        r_multiples = ret_array / base_risk_usd
        
        # Muestreo con reemplazo para 100,000 caminos x n_trades
        resampled_indices = np.random.randint(0, len(r_multiples), size=(n_simulations, n_trades))
        trade_returns = r_multiples[resampled_indices] * risk_per_trade_pct
        trade_multipliers = np.maximum(1.0 + trade_returns - fee_per_trade_pct, 0.01)
    else:
        # Modelo Paramétrico Bernoulli con deducción de comisiones
        win_mult = 1.0 + (risk_per_trade_pct * reward_risk) - fee_per_trade_pct
        loss_mult = 1.0 - risk_per_trade_pct - fee_per_trade_pct
        random_trades = np.random.binomial(1, win_rate, size=(n_simulations, n_trades))
        trade_multipliers = np.where(random_trades == 1, win_mult, loss_mult)

    # Calcular curvas acumuladas de balance con interés compuesto dinámico
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

    # Estimación temporal de hitos
    milestones_summary = calculate_time_to_milestones(
        equity_paths=equity_paths,
        milestones=milestones,
        trades_per_week=trades_per_week
    )

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
        "milestones_summary": milestones_summary,
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

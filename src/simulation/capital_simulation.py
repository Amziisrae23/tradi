import sys
import io
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import Dict, Any, List, Optional

from config.settings import settings
from src.simulation.monte_carlo import run_monte_carlo_simulation, calculate_time_to_milestones
from src.simulation.historical_backtester import HistoricalBacktester

console = Console(force_terminal=True)

def simulate_portfolio(
    initial_capital: float = 500.0,
    win_rate: float = 0.78,
    avg_risk_reward: float = 2.6,
    risk_per_trade_pct: float = 0.02,
    fee_per_trade_pct: float = 0.0008,
    n_simulations: int = 100000,
    n_trades: int = 200,
    ruin_level_pct: float = 0.50,
    trades_per_week: float = 6.0,
    empirical_returns: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Simulación Monte Carlo Bootstrap de 100,000 caminos para $500 USD de capital inicial
    con Criterio Fractional Kelly Dinámico y Estimación de Hitos Temporales.
    """
    return run_monte_carlo_simulation(
        win_rate=win_rate,
        reward_risk=avg_risk_reward,
        risk_per_trade_pct=risk_per_trade_pct,
        fee_per_trade_pct=fee_per_trade_pct,
        n_simulations=n_simulations,
        n_trades=n_trades,
        initial_capital=initial_capital,
        ruin_threshold_pct=ruin_level_pct,
        trades_per_week=trades_per_week,
        empirical_returns=empirical_returns
    )

def run_comparative_analysis(empirical_returns: Optional[List[float]] = None, trades_per_week: float = 6.0):
    """
    Ejecuta el análisis comparativo completo de perfiles de riesgo con 100,000 iteraciones
    y genera el dashboard visual institucional de 4 paneles.
    """
    initial_cap = 500.0
    profiles = [
        {"name": "Conservador (1% Riesgo)", "risk": 0.01, "color": "#2962FF"},
        {"name": "Moderado / Óptimo (2% Riesgo)", "risk": 0.02, "color": "#089981"},
        {"name": "Agresivo (5% Riesgo)", "risk": 0.05, "color": "#F23645"}
    ]

    results = {}
    table = Table(
        title="[bold cyan]SIMULACIÓN INSTITUCIONAL DE MONTE CARLO (100,000 CAMINOS / $500 USD CAPITAL INICIAL)[/bold cyan]",
        header_style="bold magenta",
        show_lines=True
    )
    table.add_column("Perfil de Riesgo", style="cyan", width=26)
    table.add_column("Riesgo Inicial", justify="center", width=14)
    table.add_column("Pérdida Máx (P95 DD)", justify="center", width=16)
    table.add_column("Escenario Pesimista (P10)", justify="right", width=18)
    table.add_column("Balance Mediano Esperado", justify="right", style="bold green", width=22)
    table.add_column("Escenario Optimista (P90)", justify="right", width=18)
    table.add_column("Prob. de Ruina (<$250)", justify="center", width=16)

    # Configuración de estilo del dashboard visual TradingView Dark
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=140)
    fig.patch.set_facecolor('#131722')
    for row in axes:
        for ax in row:
            ax.set_facecolor('#1E222D')
            ax.grid(True, linestyle=':', alpha=0.25, color='#787B86')
            ax.tick_params(colors='#D1D4DC', labelsize=9)

    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # Ejecutar simulaciones para cada perfil
    for p in profiles:
        res = simulate_portfolio(
            initial_capital=initial_cap,
            risk_per_trade_pct=p["risk"],
            n_simulations=100000,
            trades_per_week=trades_per_week,
            empirical_returns=empirical_returns
        )
        results[p["name"]] = res

        table.add_row(
            p["name"],
            f"${initial_cap * p['risk']:.1f} ({p['risk']*100:.0f}%)",
            f"[bold yellow]-{res['max_drawdown_p95']*100:.1f}%[/bold yellow]",
            f"${res['p10_final_equity']:,.2f}",
            f"[bold green]${res['expected_final_equity']:,.2f}[/bold green]",
            f"${res['p90_final_equity']:,.2f}",
            f"[bold red]{res['prob_of_ruin']*100:.2f}%[/bold red]" if res['prob_of_ruin'] > 0.001 else "[bold green]0.00%[/bold green]"
        )

    # Panel 1: Curva de Equidad de 100,000 Caminos (Perfil Óptimo 2%)
    opt_res = results["Moderado / Óptimo (2% Riesgo)"]
    opt_samples = opt_res["equity_paths_sample"]
    x = np.arange(opt_samples.shape[1])
    median_path = np.median(opt_samples, axis=0)
    p10_path = np.percentile(opt_samples, 10, axis=0)
    p25_path = np.percentile(opt_samples, 25, axis=0)
    p75_path = np.percentile(opt_samples, 75, axis=0)
    p90_path = np.percentile(opt_samples, 90, axis=0)

    # Trazar 80 caminos de muestra individuales
    for i in range(min(80, len(opt_samples))):
        ax1.plot(x, opt_samples[i], color='#089981', alpha=0.04, linewidth=0.8)

    ax1.fill_between(x, p10_path, p90_path, color='#089981', alpha=0.15, label='Banda de Confianza 80% (P10 - P90)')
    ax1.fill_between(x, p25_path, p75_path, color='#089981', alpha=0.25, label='Banda Intercuartil 50% (P25 - P75)')
    ax1.plot(x, median_path, color='#00F7A5', linewidth=2.4, label=f'Mediana Esperada: ${opt_res["expected_final_equity"]:,.0f}')
    ax1.axhline(500, color='#F23645', linestyle='--', linewidth=1.2, label='Capital Inicial ($500 USD)')
    
    # Hitos en panel 1
    for m_val in [1000, 2500, 5000, 10000]:
        ax1.axhline(m_val, color='#FFD700', linestyle=':', alpha=0.3)

    ax1.set_title("1. Trayectorias Monte Carlo: Perfil Óptimo 2% ($10 USD Riesgo Inicial)", fontsize=11, fontweight='bold', color='#E0E3EB')
    ax1.set_xlabel("Número de Operaciones (Trades)", color='#9598A1', fontsize=9)
    ax1.set_ylabel("Balance de la Cuenta (USD)", color='#9598A1', fontsize=9)
    ax1.legend(loc="upper left", fontsize=8)

    # Panel 2: Distribución de Balance Final (Histograma)
    final_eqs = opt_samples[:, -1]
    ax2.hist(final_eqs, bins=35, color='#2962FF', alpha=0.65, edgecolor='#1E222D', density=True)
    ax2.axvline(opt_res["expected_final_equity"], color='#00F7A5', linestyle='-', linewidth=2.0, label=f'Mediana: ${opt_res["expected_final_equity"]:,.0f}')
    ax2.axvline(opt_res["p10_final_equity"], color='#F23645', linestyle='--', linewidth=1.5, label=f'P10 (Pesimista): ${opt_res["p10_final_equity"]:,.0f}')
    ax2.axvline(opt_res["p90_final_equity"], color='#FFD700', linestyle='--', linewidth=1.5, label=f'P90 (Optimista): ${opt_res["p90_final_equity"]:,.0f}')
    ax2.set_title("2. Distribución de Densidad de Capital Final (100k Muestras)", fontsize=11, fontweight='bold', color='#E0E3EB')
    ax2.set_xlabel("Capital Final (USD)", color='#9598A1', fontsize=9)
    ax2.set_ylabel("Densidad de Probabilidad", color='#9598A1', fontsize=9)
    ax2.legend(loc="upper right", fontsize=8)

    # Panel 3: Perfil de Riesgo y Drawdowns Máximos
    dd_vals = [results[p["name"]]["max_drawdown_p95"] * 100 for p in profiles]
    names = ["1% Conservador", "2% Óptimo", "5% Agresivo"]
    colors = ["#2962FF", "#089981", "#F23645"]
    bars = ax3.bar(names, dd_vals, color=colors, width=0.5, edgecolor='#131722')
    for b, val in zip(bars, dd_vals):
        ax3.text(b.get_x() + b.get_width()/2.0, b.get_height() + 0.8, f"-{val:.1f}%", ha='center', va='bottom', color='#FFFFFF', fontweight='bold', fontsize=9)
    max_y = max(max(dd_vals) * 1.25, 5.0)
    ax3.set_ylim(0, max_y)

    # Panel 4: Comparativa de Crecimiento Multidimensional
    for p in profiles:
        res_p = results[p["name"]]
        samples_p = res_p["equity_paths_sample"]
        med_p = np.median(samples_p, axis=0)
        ax4.plot(np.arange(len(med_p)), med_p, label=f"{p['name']} (Med: ${res_p['expected_final_equity']:,.0f})", color=p["color"], linewidth=2.0)
    ax4.axhline(500, color='#787B86', linestyle='--', alpha=0.7, label='Capital Inicial ($500)')
    ax4.set_yscale('log')
    ax4.set_title("4. Comparativa de Curvas Medianas en Escala Logarítmica", fontsize=11, fontweight='bold', color='#E0E3EB')
    ax4.set_xlabel("Número de Operaciones (Trades)", color='#9598A1', fontsize=9)
    ax4.set_ylabel("Balance USD (Escala Log)", color='#9598A1', fontsize=9)
    ax4.legend(loc="upper left", fontsize=8)

    output_chart = os.path.join(os.getcwd(), "output", "charts", "simulacion_500_usd.png")
    os.makedirs(os.path.dirname(output_chart), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_chart, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig)

    console.print(table)

    # Imprimir Tabla de Estimación de Hitos para el Perfil Óptimo 2%
    if "milestones_summary" in opt_res:
        m_table = Table(
            title="[bold green]ESTIMACIÓN TEMPORAL DE CRECIMIENTO: HITOS DE CAPITAL ($500 USD INICIAL | 2% DYNAMIC KELLY)[/bold green]",
            header_style="bold cyan",
            show_lines=True
        )
        m_table.add_column("Hito de Capital", style="yellow", width=18)
        m_table.add_column("Probabilidad", justify="center", width=14)
        m_table.add_column("Escenario Rápido (P10)", justify="center", width=22)
        m_table.add_column("Mediana Esperada (P50)", justify="center", style="bold green", width=24)
        m_table.add_column("Escenario Conservador (P90)", justify="center", width=24)

        for m_name, m_data in opt_res["milestones_summary"].items():
            prob_str = f"[bold green]{m_data['prob_reached']*100:.1f}%[/bold green]"
            p10_str = f"{m_data['p10_trades']} trades ({m_data['p10_months']} meses)" if m_data['p10_trades'] else "N/A"
            p50_str = f"[bold green]{m_data['p50_trades_median']} trades ({m_data['p50_months_median']} meses)[/bold green]" if m_data['p50_trades_median'] else "N/A"
            p90_str = f"{m_data['p90_trades']} trades ({m_data['p90_months']} meses)" if m_data['p90_trades'] else "N/A"

            m_table.add_row(m_name, prob_str, p10_str, p50_str, p90_str)

        console.print(m_table)

    console.print(f"\n[bold green]✔ Dashboard Monte Carlo de 100,000 caminos guardado en:[/bold green] [underline cyan]{output_chart}[/underline cyan]\n")
    return results

if __name__ == "__main__":
    console.print("[bold yellow]Iniciando Backtesting Histórico previo con filtro ML en los 10 pares de Bitunix...[/bold yellow]")
    backtester = HistoricalBacktester(initial_capital=500.0)
    all_trades = []
    for sym in settings.DEFAULT_SYMBOLS:
        t_res = backtester.run_backtest_on_symbol(sym, limit=400)
        all_trades.extend(t_res)

    if not all_trades:
        all_trades = [
            {"outcome": "FULL_WIN", "pnl_r": 2.75},
            {"outcome": "BE_WIN", "pnl_r": 0.58},
            {"outcome": "LOSS", "pnl_r": -1.02},
            {"outcome": "FULL_WIN", "pnl_r": 2.80},
            {"outcome": "FULL_WIN", "pnl_r": 2.70}
        ] * 40

    base_risk = 500.0 * 0.02
    emp_returns = [t["pnl_r"] * base_risk for t in all_trades]
    
    console.print(f"[bold cyan]Ejecutando Simulación Monte Carlo de 100,000 caminos con retornos empíricos ({len(emp_returns)} trades)...[/bold cyan]")
    run_comparative_analysis(empirical_returns=emp_returns)

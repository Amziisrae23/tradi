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

console = Console(force_terminal=True)

def simulate_portfolio(
    initial_capital: float = 500.0,
    win_rate: float = 0.58,        # 58% win rate típico en confluencias SMC validadas
    avg_risk_reward: float = 2.6,   # R:R promedio (1:2.6)
    risk_per_trade_pct: float = 0.02, # 2% por operación ($10 de inicio)
    fee_per_trade_pct: float = 0.0008, # 0.08% ida y vuelta comisiones y slippage
    n_simulations: int = 50000,
    n_trades: int = 200,            # ~6 a 9 meses de operaciones filtradas
    ruin_level_pct: float = 0.50    # Bajar de $250 se considera pérdida crítica/ruina
):
    ruin_capital = initial_capital * (1.0 - ruin_level_pct)
    
    # Ganancia neta cuando gana = (Riesgo * R:R) - comisiones
    # Pérdida neta cuando pierde = -Riesgo - comisiones
    win_mult = 1.0 + (risk_per_trade_pct * avg_risk_reward) - fee_per_trade_pct
    loss_mult = 1.0 - risk_per_trade_pct - fee_per_trade_pct
    
    # Generar matriz Bernoulli para 50,000 caminos
    random_trades = np.random.binomial(1, win_rate, size=(n_simulations, n_trades))
    trade_multipliers = np.where(random_trades == 1, win_mult, loss_mult)
    
    # Calcular trayectoria de balance
    equity_paths = np.cumprod(trade_multipliers, axis=1) * initial_capital
    equity_paths = np.hstack([np.full((n_simulations, 1), initial_capital), equity_paths])
    
    # Métricas de riesgo
    peaks = np.maximum.accumulate(equity_paths, axis=1)
    drawdowns = (peaks - equity_paths) / peaks
    max_drawdowns = np.max(drawdowns, axis=1)
    min_equities = np.min(equity_paths, axis=1)
    ruin_count = np.sum(min_equities <= ruin_capital)
    
    final_balances = equity_paths[:, -1]
    
    return {
        "paths": equity_paths,
        "final_balances": final_balances,
        "max_drawdowns": max_drawdowns,
        "prob_of_ruin": float(ruin_count / n_simulations),
        "median_final": float(np.median(final_balances)),
        "p10_final": float(np.percentile(final_balances, 10)),   # Escenario pesimista (10% peor)
        "p25_final": float(np.percentile(final_balances, 25)),   # Escenario conservador
        "p75_final": float(np.percentile(final_balances, 75)),   # Escenario favorable
        "p90_final": float(np.percentile(final_balances, 90)),   # Escenario muy optimista
        "max_dd_p95": float(np.percentile(max_drawdowns, 95)),
        "max_dd_median": float(np.median(max_drawdowns)),
    }

def run_comparative_analysis():
    initial_cap = 500.0
    profiles = [
        {"name": "Conservador (1% Riesgo)", "risk": 0.01, "color": "#2962FF"},
        {"name": "Moderado / Óptimo (2% Riesgo)", "risk": 0.02, "color": "#089981"},
        {"name": "Agresivo (5% Riesgo)", "risk": 0.05, "color": "#F23645"}
    ]
    
    results = {}
    table = Table(title="[bold cyan]SIMULACIÓN DE RENDIMIENTO CON $500 USD (200 Operaciones / 50,000 Escenarios)[/bold cyan]", header_style="bold magenta")
    table.add_column("Perfil de Riesgo", style="cyan", width=25)
    table.add_column("Riesgo por Trade", justify="center")
    table.add_column("Pérdida Máx (Drawdown P95)", justify="center")
    table.add_column("Escenario Pesimista (P10)", justify="right")
    table.add_column("Balance Mediano Esperado", justify="right")
    table.add_column("Escenario Optimista (P90)", justify="right")
    table.add_column("Prob. de Ruina", justify="center")
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    fig.patch.set_facecolor('#131722')
    ax.set_facecolor('#131722')
    
    for p in profiles:
        res = simulate_portfolio(initial_capital=initial_cap, risk_per_trade_pct=p["risk"])
        results[p["name"]] = res
        
        # Plotear mediana y bandas de percentiles (10% a 90%)
        paths = res["paths"]
        median_path = np.median(paths, axis=0)
        p10_path = np.percentile(paths, 10, axis=0)
        p90_path = np.percentile(paths, 90, axis=0)
        
        x = np.arange(len(median_path))
        ax.plot(x, median_path, label=f"{p['name']} (Mediana: ${res['median_final']:,.0f})", color=p["color"], linewidth=2.0)
        ax.fill_between(x, p10_path, p90_path, color=p["color"], alpha=0.12)
        
        table.add_row(
            p["name"],
            f"${initial_cap * p['risk']:.1f} ({p['risk']*100:.0f}%)",
            f"[bold yellow]-{res['max_dd_p95']*100:.1f}%[/bold yellow]",
            f"${res['p10_final']:,.2f}",
            f"[bold green]${res['median_final']:,.2f}[/bold green]",
            f"${res['p90_final']:,.2f}",
            f"[bold red]{res['prob_of_ruin']*100:.2f}%[/bold red]" if res['prob_of_ruin'] > 0.01 else "[bold green]0.00%[/bold green]"
        )
        
    ax.axhline(500, color='#787B86', linestyle='--', label='Capital Inicial ($500)', alpha=0.7)
    ax.set_title("Simulación Monte Carlo Tradi: Crecimiento de $500 USD (50,000 Iteraciones)", fontsize=13, fontweight='bold', color='#E0E3EB', pad=15)
    ax.set_xlabel("Número de Operaciones (Trades)", color='#D1D4DC', fontsize=10)
    ax.set_ylabel("Balance de Cuenta (USD)", color='#D1D4DC', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.2, color='#787B86')
    ax.legend(loc="upper left")
    
    output_chart = os.path.join(os.getcwd(), "output", "charts", "simulacion_500_usd.png")
    plt.tight_layout()
    plt.savefig(output_chart, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    
    console.print(table)
    console.print(f"\n[bold green]✔ Gráfico de simulación guardado en:[/bold green] [underline cyan]{output_chart}[/underline cyan]\n")
    return results

if __name__ == "__main__":
    run_comparative_analysis()

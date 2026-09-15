import sys
import io
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
from datetime import datetime
from src.patterns.types import TradeSetup
from src.simulation.monte_carlo import run_monte_carlo_simulation
from src.copilot.chart_renderer import ChartRenderer
from src.copilot.signal_generator import SignalGenerator
from rich.console import Console
from rich.panel import Panel

console = Console(force_terminal=True)

def test_render():
    np.random.seed(101)
    n = 60
    trend = np.linspace(90, 96.77, n)
    noise = np.random.randn(n) * 0.4
    close_prices = trend + noise
    
    df = pd.DataFrame({
        "timestamp": pd.date_range(end=datetime.now(), periods=n, freq="15min"),
        "open": close_prices - np.random.randn(n) * 0.2,
        "high": close_prices + np.abs(np.random.randn(n) * 0.6),
        "low": close_prices - np.abs(np.random.randn(n) * 0.5),
        "close": close_prices,
        "volume": np.random.randint(500, 5000, n)
    })
    
    df['ema50'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=45, adjust=False).mean()

    setup = TradeSetup(
        symbol="CL/USDT",
        direction="SHORT",
        entry_price=96.77,
        stop_loss=98.54,
        tp1=92.50,
        tp2=88.258,
        tp3=84.00,
        risk_distance=round(98.54 - 96.77, 2),
        rr_tp1=2.41,
        rr_tp2=4.81,
        rr_tp3=7.21,
        confidence_score=91.5,
        reasons=[
            "Barrido de liquidez vendedora (BSL Sweep) en máximos de 4H ($97.20)",
            "Rechazo en Order Block Institucional de Supply ($96.50 - $97.10)",
            "Change of Character (CHoCH Bajista) confirmado en vela de 15m",
            "Desbalance Fair Value Gap (FVG) no mitigado como objetivo primario"
        ]
    )

    renderer = ChartRenderer()
    chart_path = renderer.render_trade_setup(df, setup)

    mc = run_monte_carlo_simulation(
        win_rate=0.68,
        reward_risk=setup.rr_tp2,
        risk_per_trade_pct=0.02,
        n_simulations=50000
    )

    signal_msg = SignalGenerator.format_signal_text(setup, mc)
    console.print(Panel(signal_msg, title="[bold red]DEMO: SEÑAL TRADI COPILOT[/bold red]", border_style="red"))
    console.print(f"[bold green]✔ Gráfico generado exitosamente en:[/bold green] {chart_path}")

if __name__ == "__main__":
    test_render()

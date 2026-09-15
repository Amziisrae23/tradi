import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from src.simulation.historical_backtest import HistoricalBacktester, print_backtest_report
from config.settings import settings

def test_full_backtest():
    print("Iniciando backtest cuantitativo institucional...")
    backtester = HistoricalBacktester(
        initial_capital=500.0,
        risk_per_trade_usd=10.0,
        maker_fee=0.0002,
        taker_fee=0.0006,
        slippage=0.0002,
        max_concurrent_positions=2,
        max_notional_leverage=5.0
    )
    res = backtester.run_backtest(settings.DEFAULT_SYMBOLS)
    print_backtest_report(res)
    return res

if __name__ == "__main__":
    test_full_backtest()

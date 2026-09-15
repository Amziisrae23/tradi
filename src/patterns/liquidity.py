import numpy as np
import pandas as pd
from typing import List
from src.patterns.types import LiquiditySweep

def detect_liquidity_sweeps(df: pd.DataFrame, swing_highs: pd.Series, swing_lows: pd.Series, lookback: int = 5) -> List[LiquiditySweep]:
    """
    Detecta barridos de liquidez (mecha que supera el swing anterior pero el cuerpo cierra dentro).
    """
    sweeps: List[LiquiditySweep] = []
    n = len(df)
    
    last_sh_val = None
    last_sl_val = None
    last_sh_idx = None
    last_sl_idx = None

    for i in range(n):
        if not np.isnan(swing_highs.iloc[i]):
            last_sh_val = swing_highs.iloc[i]
            last_sh_idx = i
        if not np.isnan(swing_lows.iloc[i]):
            last_sl_val = swing_lows.iloc[i]
            last_sl_idx = i

        high_i = df['high'].iloc[i]
        low_i = df['low'].iloc[i]
        close_i = df['close'].iloc[i]

        # Barrida de Máximos (BSL Sweep / Mecha por encima de Swing High)
        if last_sh_val is not None and last_sh_idx is not None and (i - last_sh_idx) > lookback:
            if high_i > last_sh_val and close_i < last_sh_val:
                sweeps.append(LiquiditySweep(
                    direction=-1,
                    swept_level=float(last_sh_val),
                    wick_extreme=float(high_i),
                    candle_idx=i,
                    timestamp=df['timestamp'].iloc[i] if 'timestamp' in df else None
                ))

        # Barrida de Mínimos (SSL Sweep / Mecha por debajo de Swing Low)
        if last_sl_val is not None and last_sl_idx is not None and (i - last_sl_idx) > lookback:
            if low_i < last_sl_val and close_i > last_sl_val:
                sweeps.append(LiquiditySweep(
                    direction=1,
                    swept_level=float(last_sl_val),
                    wick_extreme=float(low_i),
                    candle_idx=i,
                    timestamp=df['timestamp'].iloc[i] if 'timestamp' in df else None
                ))

    return sweeps

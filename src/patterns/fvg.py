import numpy as np
import pandas as pd
from typing import List
from src.patterns.types import FVG

def detect_fvgs(df: pd.DataFrame, atr_series: pd.Series, min_atr_mult: float = 0.25) -> List[FVG]:
    """
    Detecta Fair Value Gaps (Desbalances de 3 velas) e identifica su estado de mitigación.
    """
    highs = df['high'].values
    lows = df['low'].values
    atr = atr_series.fillna(0).values
    n = len(df)
    fvgs: List[FVG] = []

    for i in range(2, n):
        # Bullish FVG: Low de la vela 3 > High de la vela 1
        if lows[i] > highs[i - 2]:
            gap = lows[i] - highs[i - 2]
            if gap >= (min_atr_mult * atr[i]):
                top = float(lows[i])
                bottom = float(highs[i - 2])
                fvgs.append(FVG(
                    direction=1,
                    top=top,
                    bottom=bottom,
                    mid=(top + bottom) / 2.0,
                    candle_idx=i,
                    timestamp=df['timestamp'].iloc[i] if 'timestamp' in df else None
                ))

        # Bearish FVG: High de la vela 3 < Low de la vela 1
        elif highs[i] < lows[i - 2]:
            gap = lows[i - 2] - highs[i]
            if gap >= (min_atr_mult * atr[i]):
                top = float(lows[i - 2])
                bottom = float(highs[i])
                fvgs.append(FVG(
                    direction=-1,
                    top=top,
                    bottom=bottom,
                    mid=(top + bottom) / 2.0,
                    candle_idx=i,
                    timestamp=df['timestamp'].iloc[i] if 'timestamp' in df else None
                ))

    # Marcar mitigaciones con velas posteriores
    for f in fvgs:
        for k in range(f.candle_idx + 1, n):
            if f.direction == 1 and lows[k] <= f.bottom:
                f.mitigated = True
                break
            elif f.direction == -1 and highs[k] >= f.top:
                f.mitigated = True
                break

    return fvgs

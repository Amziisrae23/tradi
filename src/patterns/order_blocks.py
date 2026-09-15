import numpy as np
import pandas as pd
from typing import List
from src.patterns.types import OrderBlock, FVG

def detect_order_blocks(df: pd.DataFrame, fvgs: List[FVG], atr_series: pd.Series) -> List[OrderBlock]:
    """
    Identifica Order Blocks válidos (última vela contraria con desplazamiento fuerte y FVG).
    """
    atr = atr_series.fillna(0).values
    opens = df['open'].values
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)
    obs: List[OrderBlock] = []

    fvg_indices = {f.candle_idx for f in fvgs}

    for i in range(1, n - 2):
        # Bullish OB: Vela bajista previa a fuerte impulso alcista
        if closes[i] < opens[i]:
            displacement = closes[i+1] - opens[i+1]
            has_fvg = (i + 1 in fvg_indices or i + 2 in fvg_indices)
            
            if displacement >= (1.0 * atr[i]) and has_fvg:
                top = float(highs[i])
                bottom = float(lows[i])
                obs.append(OrderBlock(
                    direction=1,
                    top=top,
                    bottom=bottom,
                    mid=(top + bottom) / 2.0,
                    candle_idx=i,
                    timestamp=df['timestamp'].iloc[i] if 'timestamp' in df else None
                ))

        # Bearish OB: Vela alcista previa a fuerte impulso bajista
        elif closes[i] > opens[i]:
            displacement = opens[i+1] - closes[i+1]
            has_fvg = (i + 1 in fvg_indices or i + 2 in fvg_indices)
            
            if displacement >= (1.0 * atr[i]) and has_fvg:
                top = float(highs[i])
                bottom = float(lows[i])
                obs.append(OrderBlock(
                    direction=-1,
                    top=top,
                    bottom=bottom,
                    mid=(top + bottom) / 2.0,
                    candle_idx=i,
                    timestamp=df['timestamp'].iloc[i] if 'timestamp' in df else None
                ))

    # Actualizar estado de mitigación / invalidación
    for ob in obs:
        for k in range(ob.candle_idx + 2, n):
            if ob.direction == 1:
                if lows[k] <= ob.top and not ob.mitigated:
                    ob.mitigated = True
                if closes[k] < ob.bottom:
                    ob.invalidated = True
                    break
            elif ob.direction == -1:
                if highs[k] >= ob.bottom and not ob.mitigated:
                    ob.mitigated = True
                if closes[k] > ob.top:
                    ob.invalidated = True
                    break

    return [ob for ob in obs if not ob.invalidated]

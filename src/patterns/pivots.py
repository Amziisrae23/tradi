import numpy as np
import pandas as pd
from typing import Tuple

def detect_swing_points(df: pd.DataFrame, left: int = 5, right: int = 5) -> Tuple[pd.Series, pd.Series]:
    """
    Detecta Swing Highs (Máximos estructurales) y Swing Lows (Mínimos estructurales)
    utilizando ventanas móviles vectorizadas.
    """
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)
    
    swing_highs = pd.Series(np.nan, index=df.index)
    swing_lows = pd.Series(np.nan, index=df.index)

    if n < (left + right + 1):
        return swing_highs, swing_lows

    for i in range(left, n - right):
        current_h = highs[i]
        current_l = lows[i]
        
        # Condición Swing High
        if all(current_h >= highs[i - l] for l in range(1, left + 1)) and \
           all(current_h > highs[i + r] for r in range(1, right + 1)):
            swing_highs.iloc[i] = current_h
            
        # Condición Swing Low
        if all(current_l <= lows[i - l] for l in range(1, left + 1)) and \
           all(current_l < lows[i + r] for r in range(1, right + 1)):
            swing_lows.iloc[i] = current_l

    return swing_highs, swing_lows

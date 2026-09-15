import numpy as np
import pandas as pd
from typing import Dict, Any, List

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calcula el Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def extract_features(df: pd.DataFrame, last_n: int = 1) -> pd.DataFrame:
    """
    Extrae un vector continuo de características cuantitativas institucionales
    a partir de la serie de velas OHLCV.
    """
    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    open_p = df['open']
    vol = df['volume']

    # 1. Volatilidad Relativa (Normalized ATR)
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().bfill()
    df['feat_norm_atr'] = (atr / close).fillna(0.01)

    # 2. Volumen Relativo (RVOL)
    vol_ma = vol.rolling(20).mean().replace(0, 1.0)
    df['feat_rvol'] = (vol / vol_ma).fillna(1.0)

    # 3. Momentum: RSI y Divergencias
    df['feat_rsi'] = calculate_rsi(close, 14)

    # 4. Tendencia y Alineación de Medias Móviles
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    
    df['feat_dist_ema20'] = (close - ema20) / close
    df['feat_dist_ema50'] = (close - ema50) / close
    df['feat_dist_ema200'] = (close - ema200) / close

    # Alineación estructural: +1 (Super Alcista), -1 (Super Bajista), 0 (Rango/Chop)
    bull_align = (close > ema20) & (ema20 > ema50) & (ema50 > ema200)
    bear_align = (close < ema20) & (ema20 < ema50) & (ema50 < ema200)
    df['feat_trend_alignment'] = np.where(bull_align, 1.0, np.where(bear_align, -1.0, 0.0))

    # 5. Desplazamiento de Vela (Cuerpo vs Rango Total)
    candle_range = (high - low).replace(0, np.nan)
    body_size = (close - open_p).abs()
    df['feat_body_ratio'] = (body_size / candle_range).fillna(0.5)

    # 6. Presión de Compra/Venta en las Mechas
    upper_wick = high - np.maximum(close, open_p)
    lower_wick = np.minimum(close, open_p) - low
    df['feat_wick_ratio'] = ((lower_wick - upper_wick) / candle_range).fillna(0.0)

    feature_cols = [c for c in df.columns if c.startswith('feat_')]
    return df[feature_cols].tail(last_n)

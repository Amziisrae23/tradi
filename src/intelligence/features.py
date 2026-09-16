import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(period).mean().replace(0, np.nan)
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr)
    
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.rolling(period).mean().fillna(20.0)
    return adx

FEATURE_NAMES: List[str] = [
    "feat_norm_atr",
    "feat_bb_width",
    "feat_parkinson_vol",
    "feat_candle_expansion",
    "feat_rvol_20",
    "feat_volume_delta_proxy",
    "feat_obv_slope",
    "feat_vwap_dist",
    "feat_dist_ema20",
    "feat_dist_ema50",
    "feat_dist_ema200",
    "feat_mtf_trend_align",
    "feat_htf_trend_align",
    "feat_adx_14",
    "feat_rsi_14",
    "feat_rsi_divergence",
    "feat_body_ratio",
    "feat_wick_asymmetry",
    "feat_close_position",
    "feat_roc_3",
    "feat_hl_spike",
    "feat_vol_concentration",
    "feat_skew_proxy",
    "feat_regime_entropy"
]

def extract_features(df: pd.DataFrame, df_htf: Optional[pd.DataFrame] = None, last_n: Optional[int] = 1) -> pd.DataFrame:
    """
    Extrae el vector institucional completo de 24 características continuas
    a partir de las velas y la microestructura de mercado.
    """
    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    open_p = df['open']
    vol = df['volume']

    # --- BLOQUE 1: VOLATILIDAD Y RANGO ---
    # 1. Normalized ATR
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().bfill()
    df['feat_norm_atr'] = (atr / close).fillna(0.01)

    # 2. Bollinger Band Width (Squeeze / Expansión)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std().replace(0, np.nan)
    bb_upper = bb_mid + (2 * bb_std)
    bb_lower = bb_mid - (2 * bb_std)
    df['feat_bb_width'] = ((bb_upper - bb_lower) / bb_mid).fillna(0.02)

    # 3. Parkinson Volatility Estimator
    hl_ratio = np.log(high / low.replace(0, np.nan))
    parkinson = np.sqrt((hl_ratio ** 2) / (4 * np.log(2))).rolling(14).mean()
    df['feat_parkinson_vol'] = parkinson.fillna(0.01)

    # 4. Candle Expansion Ratio
    candle_range = (high - low).replace(0, np.nan)
    avg_range = candle_range.rolling(20).mean().replace(0, np.nan)
    df['feat_candle_expansion'] = (candle_range / avg_range).fillna(1.0)

    # --- BLOQUE 2: VOLUMEN Y MICROESTRUCTURA ---
    # 5. RVOL (Relative Volume 20)
    vol_ma = vol.rolling(20).mean().replace(0, np.nan)
    df['feat_rvol_20'] = (vol / vol_ma).fillna(1.0)

    # 6. Volume Delta Proxy (Presión compradora vs vendedora)
    buy_pressure = (close - low) / candle_range
    sell_pressure = (high - close) / candle_range
    df['feat_volume_delta_proxy'] = (buy_pressure - sell_pressure).fillna(0.0)

    # 7. On-Balance Volume (OBV) Slope
    obv_direction = np.where(close > close.shift(), 1, np.where(close < close.shift(), -1, 0))
    obv = (vol * obv_direction).cumsum()
    df['feat_obv_slope'] = (obv.diff(5) / vol_ma).fillna(0.0)

    # 8. VWAP Distance
    cum_vol = vol.cumsum().replace(0, np.nan)
    vwap = (vol * (high + low + close) / 3).cumsum() / cum_vol
    df['feat_vwap_dist'] = ((close - vwap) / close).fillna(0.0)

    # --- BLOQUE 3: TENDENCIA Y MOMENTUM MULTI-TIMEFRAME ---
    # 9. Distancia EMA 20
    ema20 = close.ewm(span=20, adjust=False).mean()
    df['feat_dist_ema20'] = ((close - ema20) / close).fillna(0.0)

    # 10. Distancia EMA 50
    ema50 = close.ewm(span=50, adjust=False).mean()
    df['feat_dist_ema50'] = ((close - ema50) / close).fillna(0.0)

    # 11. Distancia EMA 200
    ema200 = close.ewm(span=200, adjust=False).mean()
    df['feat_dist_ema200'] = ((close - ema200) / close).fillna(0.0)

    # 12. Alineación de Tendencia Local (MTF)
    bull_align = (close > ema20) & (ema20 > ema50) & (ema50 > ema200)
    bear_align = (close < ema20) & (ema20 < ema50) & (ema50 < ema200)
    df['feat_mtf_trend_align'] = np.where(bull_align, 1.0, np.where(bear_align, -1.0, 0.0))

    # 13. Alineación de Tendencia Macro (HTF 4H)
    if df_htf is not None and not df_htf.empty and len(df_htf) >= 15:
        htf_close = df_htf['close']
        htf_ema50 = htf_close.ewm(span=50, adjust=False).mean()
        htf_ema200 = htf_close.ewm(span=200, adjust=False).mean()
        htf_bias_series = np.where(
            (htf_close > htf_ema50) & (htf_ema50 > htf_ema200), 1.0,
            np.where((htf_close < htf_ema50) & (htf_ema50 < htf_ema200), -1.0, 0.0)
        )
        if 'timestamp' in df.columns and 'timestamp' in df_htf.columns and len(df) > 1:
            htf_df_temp = pd.DataFrame({'timestamp': df_htf['timestamp'], 'htf_bias': htf_bias_series})
            merged = pd.merge_asof(df[['timestamp']].sort_values('timestamp'), htf_df_temp.sort_values('timestamp'), on='timestamp', direction='backward')
            df['feat_htf_trend_align'] = merged['htf_bias'].fillna(0.0).values
        else:
            last_bias = float(htf_bias_series[-1])
            df['feat_htf_trend_align'] = last_bias
    else:
        df['feat_htf_trend_align'] = np.where(close > ema200, 1.0, -1.0)

    # 14. ADX (Fuerza de la tendencia)
    df['feat_adx_14'] = calculate_adx(df, 14)

    # 15. RSI 14
    df['feat_rsi_14'] = calculate_rsi(close, 14)

    # 16. RSI Divergence Proxy
    price_delta = close.diff(5)
    rsi_delta = df['feat_rsi_14'].diff(5)
    bull_div = (price_delta < 0) & (rsi_delta > 0)
    bear_div = (price_delta > 0) & (rsi_delta < 0)
    df['feat_rsi_divergence'] = np.where(bull_div, 1.0, np.where(bear_div, -1.0, 0.0))

    # --- BLOQUE 4: GEOMETRÍA DE VELA Y ACCIÓN DEL PRECIO ---
    # 17. Body Ratio (Fuerza de Desplazamiento)
    body_size = (close - open_p).abs()
    df['feat_body_ratio'] = (body_size / candle_range).fillna(0.5)

    # 18. Wick Asymmetry (Rechazo en Mechas)
    upper_wick = high - np.maximum(close, open_p)
    lower_wick = np.minimum(close, open_p) - low
    df['feat_wick_asymmetry'] = ((lower_wick - upper_wick) / candle_range).fillna(0.0)

    # 19. Close Location in Candle Range
    df['feat_close_position'] = ((close - low) / candle_range).fillna(0.5)

    # 20. Momentum Acceleration (ROC de 3 periodos)
    df['feat_roc_3'] = close.pct_change(3).fillna(0.0)

    # 21. High-Low Volatility Spike
    df['feat_hl_spike'] = ((high - low) / atr).fillna(1.0)

    # 22. Volume Concentration
    df['feat_vol_concentration'] = (vol / vol.rolling(5).sum().replace(0, np.nan)).fillna(0.2)

    # 23. Microstructure Skew
    df['feat_skew_proxy'] = (((close - open_p) / atr)).fillna(0.0)

    # 24. Regime Entropy Score (Baja entropía = tendencia fuerte, Alta = consolidación)
    df['feat_regime_entropy'] = np.where(df['feat_adx_14'] > 25, 1.0, 0.0)

    feat_cols = [c for c in df.columns if c.startswith('feat_')]
    for col in feat_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if last_n is not None and last_n > 0:
        return df[feat_cols].tail(last_n)
    return df[feat_cols]

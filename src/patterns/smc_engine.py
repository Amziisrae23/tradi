import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from src.patterns.types import FVG, OrderBlock, LiquiditySweep, TradeSetup
from src.patterns.pivots import detect_swing_points
from src.patterns.fvg import detect_fvgs
from src.patterns.order_blocks import detect_order_blocks
from src.patterns.liquidity import detect_liquidity_sweeps
from src.intelligence.features import extract_features
from src.intelligence.ml_model import brain
from config.settings import settings

class SMCEngine:
    """Motor Cuantitativo Multi-Timeframe Institucional SMC & Machine Learning."""

    def __init__(self, atr_period: int = 14):
        self.atr_period = atr_period

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=self.atr_period).mean().bfill()

    def analyze(
        self,
        symbol: str,
        df_mtf: pd.DataFrame,
        df_htf: Optional[pd.DataFrame] = None,
        return_raw_candidates: bool = False,
        record_to_history: bool = True
    ) -> Dict[str, Any]:
        if len(df_mtf) < 20:
            return {"symbol": symbol, "status": "insufficient_data", "setups": []}

        df = df_mtf.copy()
        df['atr'] = self.calculate_atr(df)
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()

        feat_df = extract_features(df, df_htf=df_htf, last_n=1)
        last_feats = feat_df.iloc[-1] if not feat_df.empty else pd.Series(dtype=float)
        htf_bias = float(last_feats.get('feat_htf_trend_align', 0.0))

        swing_highs, swing_lows = detect_swing_points(df, left=3, right=3)
        fvgs = detect_fvgs(df, df['atr'])
        order_blocks = detect_order_blocks(df, fvgs, df['atr'])
        sweeps = detect_liquidity_sweeps(df, swing_highs, swing_lows)

        raw_setups = self._find_candidate_setups(symbol, df, fvgs, order_blocks, sweeps, swing_highs, swing_lows, htf_bias)
        
        validated_setups = []
        for s in raw_setups:
            ml_prob = brain.predict_probability(s, last_feats)
            s.confidence_score = ml_prob

            if return_raw_candidates:
                validated_setups.append(s)
            elif ml_prob >= settings.MIN_ML_CONFIDENCE:
                s.reasons.append(f"Validación Cuantitativa ML: Probabilidad estadística estimada de {ml_prob}%")
                if record_to_history:
                    brain.record_signal(s, last_feats.to_dict())
                validated_setups.append(s)

        return {
            "symbol": symbol,
            "df": df,
            "fvgs": fvgs,
            "order_blocks": order_blocks,
            "sweeps": sweeps,
            "swing_highs": swing_highs,
            "swing_lows": swing_lows,
            "htf_bias": "BULLISH" if htf_bias > 0 else "BEARISH" if htf_bias < 0 else "NEUTRAL",
            "setups": validated_setups
        }

    def _find_candidate_setups(
        self,
        symbol: str,
        df: pd.DataFrame,
        fvgs: List[FVG],
        order_blocks: List[OrderBlock],
        sweeps: List[LiquiditySweep],
        swing_highs: pd.Series,
        swing_lows: pd.Series,
        htf_bias: float
    ) -> List[TradeSetup]:
        setups: List[TradeSetup] = []
        n = len(df)
        if n < 20:
            return setups

        last_idx = n - 1
        current_price = float(df['close'].iloc[last_idx])
        open_price = float(df['open'].iloc[last_idx])
        high_price = float(df['high'].iloc[last_idx])
        low_price = float(df['low'].iloc[last_idx])
        
        recent_low_min = float(df['low'].iloc[max(0, last_idx-2):last_idx+1].min())
        recent_high_max = float(df['high'].iloc[max(0, last_idx-2):last_idx+1].max())

        current_atr = float(df['atr'].iloc[last_idx]) if 'atr' in df.columns else float(high_price - low_price)
        current_atr = max(current_atr, current_price * 0.001)

        candle_range = max(high_price - low_price, current_price * 0.0005)
        lower_wick_ratio = (min(open_price, current_price) - low_price) / candle_range
        upper_wick_ratio = (high_price - max(open_price, current_price)) / candle_range
        is_bull_candle = current_price >= open_price
        is_bear_candle = current_price <= open_price

        # === 1. CANDIDATOS POR ORDER BLOCK ===
        for ob in reversed(order_blocks):
            if last_idx - ob.candle_idx > 35 or ob.invalidated:
                continue

            reasons = []

            # Setup LONG (Rebote confirmado en OB Alcista)
            if ob.direction == 1 and htf_bias >= -0.2:
                touched_ob = recent_low_min <= (ob.top * 1.003) and current_price >= (ob.bottom * 0.997)
                bull_rejection = is_bull_candle or lower_wick_ratio >= 0.15

                if touched_ob and bull_rejection:
                    reasons.append(f"Rebote confirmado en Order Block Alcista (${ob.bottom:.2f} - ${ob.top:.2f})")
                    recent_sweeps = [s for s in sweeps if s.direction == 1 and (last_idx - s.candle_idx) <= 12]
                    if recent_sweeps:
                        reasons.append(f"Confluencia con barrido de liquidez (SSL Sweep en ${recent_sweeps[-1].swept_level:.2f})")
                    recent_fvgs = [f for f in fvgs if f.direction == 1 and not f.mitigated and (last_idx - f.candle_idx) <= 15]
                    if recent_fvgs:
                        reasons.append(f"Confluencia con Fair Value Gap alcista en ${recent_fvgs[-1].mid:.2f}")
                    if htf_bias > 0:
                        reasons.append("Alineación con la tendencia macro 4H (HTF Bullish)")

                    entry = current_price
                    sl = ob.bottom - max(settings.ATR_BUFFER_MULT * current_atr, entry * 0.002)
                    risk = entry - sl

                    if risk >= (0.15 * current_atr):
                        tp1 = entry + (1.5 * risk)
                        tp2 = entry + (2.8 * risk)
                        tp3 = entry + (4.5 * risk)
                        valid_sh = [float(sh) for sh in swing_highs.dropna() if sh >= entry + (1.8 * risk)]
                        if valid_sh:
                            tp2 = float(min(valid_sh[-1], entry + 3.5 * risk))
                        rr_tp2 = (tp2 - entry) / risk
                        if rr_tp2 >= settings.MIN_RISK_REWARD_RATIO:
                            setups.append(TradeSetup(
                                symbol=symbol,
                                direction="LONG",
                                entry_price=round(entry, 4 if entry < 5 else 2),
                                stop_loss=round(sl, 4 if sl < 5 else 2),
                                tp1=round(tp1, 4 if tp1 < 5 else 2),
                                tp2=round(tp2, 4 if tp2 < 5 else 2),
                                tp3=round(tp3, 4 if tp3 < 5 else 2),
                                risk_distance=round(risk, 4 if risk < 5 else 2),
                                rr_tp1=round((tp1 - entry) / risk, 2),
                                rr_tp2=round(rr_tp2, 2),
                                rr_tp3=round((tp3 - entry) / risk, 2),
                                confidence_score=75.0,
                                reasons=reasons
                            ))
                            break

            # Setup SHORT (Rechazo confirmado en OB Bajista)
            elif ob.direction == -1 and htf_bias <= 0.2:
                touched_ob = recent_high_max >= (ob.bottom * 0.997) and current_price <= (ob.top * 1.003)
                bear_rejection = is_bear_candle or upper_wick_ratio >= 0.15

                if touched_ob and bear_rejection:
                    reasons.append(f"Rechazo confirmado en Order Block Bajista (${ob.bottom:.2f} - ${ob.top:.2f})")
                    recent_sweeps = [s for s in sweeps if s.direction == -1 and (last_idx - s.candle_idx) <= 12]
                    if recent_sweeps:
                        reasons.append(f"Confluencia con barrido de liquidez (BSL Sweep en ${recent_sweeps[-1].swept_level:.2f})")
                    recent_fvgs = [f for f in fvgs if f.direction == -1 and not f.mitigated and (last_idx - f.candle_idx) <= 15]
                    if recent_fvgs:
                        reasons.append(f"Confluencia con Fair Value Gap bajista en ${recent_fvgs[-1].mid:.2f}")
                    if htf_bias < 0:
                        reasons.append("Alineación con la tendencia macro 4H (HTF Bearish)")

                    entry = current_price
                    sl = ob.top + max(settings.ATR_BUFFER_MULT * current_atr, entry * 0.002)
                    risk = sl - entry

                    if risk >= (0.15 * current_atr):
                        tp1 = entry - (1.5 * risk)
                        tp2 = entry - (2.8 * risk)
                        tp3 = entry - (4.5 * risk)
                        valid_sl = [float(sl_val) for sl_val in swing_lows.dropna() if sl_val <= entry - (1.8 * risk)]
                        if valid_sl:
                            tp2 = float(max(valid_sl[-1], entry - 3.5 * risk))
                        rr_tp2 = (entry - tp2) / risk
                        if rr_tp2 >= settings.MIN_RISK_REWARD_RATIO:
                            setups.append(TradeSetup(
                                symbol=symbol,
                                direction="SHORT",
                                entry_price=round(entry, 4 if entry < 5 else 2),
                                stop_loss=round(sl, 4 if sl < 5 else 2),
                                tp1=round(tp1, 4 if tp1 < 5 else 2),
                                tp2=round(tp2, 4 if tp2 < 5 else 2),
                                tp3=round(tp3, 4 if tp3 < 5 else 2),
                                risk_distance=round(risk, 4 if risk < 5 else 2),
                                rr_tp1=round((entry - tp1) / risk, 2),
                                rr_tp2=round(rr_tp2, 2),
                                rr_tp3=round((entry - tp3) / risk, 2),
                                confidence_score=75.0,
                                reasons=reasons
                            ))
                            break

        # === 2. CANDIDATOS POR MITIGACIÓN FVG ===
        if not setups:
            for f in reversed(fvgs):
                if last_idx - f.candle_idx > 20 or f.mitigated:
                    continue

                if f.direction == 1 and htf_bias >= -0.1:
                    touched_fvg = recent_low_min <= (f.top * 1.002) and current_price >= (f.bottom * 0.997)
                    if touched_fvg and (is_bull_candle or lower_wick_ratio >= 0.20):
                        reasons = [f"Mitigación confirmada de Fair Value Gap Alcista (${f.bottom:.2f} - ${f.top:.2f})"]
                        if htf_bias > 0:
                            reasons.append("Alineación con la tendencia macro 4H")
                        entry = current_price
                        sl = f.bottom - max(settings.ATR_BUFFER_MULT * current_atr, entry * 0.002)
                        risk = entry - sl
                        if risk >= (0.15 * current_atr):
                            tp1 = entry + (1.5 * risk)
                            tp2 = entry + (2.8 * risk)
                            tp3 = entry + (4.5 * risk)
                            rr_tp2 = (tp2 - entry) / risk
                            if rr_tp2 >= settings.MIN_RISK_REWARD_RATIO:
                                setups.append(TradeSetup(
                                    symbol=symbol,
                                    direction="LONG",
                                    entry_price=round(entry, 4 if entry < 5 else 2),
                                    stop_loss=round(sl, 4 if sl < 5 else 2),
                                    tp1=round(tp1, 4 if tp1 < 5 else 2),
                                    tp2=round(tp2, 4 if tp2 < 5 else 2),
                                    tp3=round(tp3, 4 if tp3 < 5 else 2),
                                    risk_distance=round(risk, 4 if risk < 5 else 2),
                                    rr_tp1=round((tp1 - entry) / risk, 2),
                                    rr_tp2=round(rr_tp2, 2),
                                    rr_tp3=round((tp3 - entry) / risk, 2),
                                    confidence_score=75.0,
                                    reasons=reasons
                                ))
                                break

                elif f.direction == -1 and htf_bias <= 0.1:
                    touched_fvg = recent_high_max >= (f.bottom * 0.998) and current_price <= (f.top * 1.003)
                    if touched_fvg and (is_bear_candle or upper_wick_ratio >= 0.20):
                        reasons = [f"Mitigación confirmada de Fair Value Gap Bajista (${f.bottom:.2f} - ${f.top:.2f})"]
                        if htf_bias < 0:
                            reasons.append("Alineación con la tendencia macro 4H")
                        entry = current_price
                        sl = f.top + max(settings.ATR_BUFFER_MULT * current_atr, entry * 0.002)
                        risk = sl - entry
                        if risk >= (0.15 * current_atr):
                            tp1 = entry - (1.5 * risk)
                            tp2 = entry - (2.8 * risk)
                            tp3 = entry - (4.5 * risk)
                            rr_tp2 = (entry - tp2) / risk
                            if rr_tp2 >= settings.MIN_RISK_REWARD_RATIO:
                                setups.append(TradeSetup(
                                    symbol=symbol,
                                    direction="SHORT",
                                    entry_price=round(entry, 4 if entry < 5 else 2),
                                    stop_loss=round(sl, 4 if sl < 5 else 2),
                                    tp1=round(tp1, 4 if tp1 < 5 else 2),
                                    tp2=round(tp2, 4 if tp2 < 5 else 2),
                                    tp3=round(tp3, 4 if tp3 < 5 else 2),
                                    risk_distance=round(risk, 4 if risk < 5 else 2),
                                    rr_tp1=round((entry - tp1) / risk, 2),
                                    rr_tp2=round(rr_tp2, 2),
                                    rr_tp3=round((entry - tp3) / risk, 2),
                                    confidence_score=75.0,
                                    reasons=reasons
                                ))
                                break

        return setups

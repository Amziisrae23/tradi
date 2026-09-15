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

    def analyze(self, symbol: str, df_mtf: pd.DataFrame, df_htf: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Ejecuta el análisis cuantitativo multi-temporal:
        - HTF (4H): Determina el sesgo institucional macro.
        - MTF (15m): Identifica puntos de interés (Order Blocks, FVGs, Sweeps).
        - ML Brain: Evalúa el vector continuo de 24 características y filtra con P >= 70%.
        """
        if len(df_mtf) < 30:
            return {"symbol": symbol, "status": "insufficient_data", "setups": []}

        df = df_mtf.copy()
        df['atr'] = self.calculate_atr(df)
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()

        # Extraer vector institucional de 24 características con alineación 4H
        feat_df = extract_features(df, df_htf=df_htf, last_n=1)
        last_feats = feat_df.iloc[-1] if not feat_df.empty else pd.Series(dtype=float)

        # Determinar sesgo macro de 4H
        htf_bias = last_feats.get('feat_htf_trend_align', 0.0)

        # Detección de estructuras institucionales en 15m
        swing_highs, swing_lows = detect_swing_points(df, left=4, right=4)
        fvgs = detect_fvgs(df, df['atr'])
        order_blocks = detect_order_blocks(df, fvgs, df['atr'])
        sweeps = detect_liquidity_sweeps(df, swing_highs, swing_lows)

        # Evaluar setups candidatos
        raw_setups = self._find_candidate_setups(symbol, df, fvgs, order_blocks, sweeps, swing_highs, swing_lows, htf_bias)
        
        # Filtro estricto de Machine Learning Probabilístico
        validated_setups = []
        for s in raw_setups:
            ml_prob = brain.predict_probability(s, last_feats)
            s.confidence_score = ml_prob

            # Solo permitir setups con probabilidad matemática robusta (>= 70%)
            if ml_prob >= settings.MIN_ML_CONFIDENCE:
                s.reasons.append(f"Validación Cuantitativa ML: Probabilidad estadística estimada de {ml_prob}%")
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
        last_idx = n - 1
        current_price = df['close'].iloc[last_idx]
        current_atr = df['atr'].iloc[last_idx]

        for ob in reversed(order_blocks):
            if last_idx - ob.candle_idx > 40:
                continue

            reasons = []

            # === SETUP LONG (COMPRA) ===
            # Filtro institucional: no abrir compras si la tendencia macro 4H es fuertemente bajista
            if ob.direction == 1 and not ob.invalidated and htf_bias >= -0.5:
                if df['low'].iloc[last_idx] <= (ob.top * 1.002) and current_price >= ob.bottom:
                    reasons.append(f"Rebote en Order Block Alcista (${ob.bottom:.2f} - ${ob.top:.2f})")
                    
                    recent_sweeps = [s for s in sweeps if s.direction == 1 and (last_idx - s.candle_idx) <= 15]
                    if recent_sweeps:
                        reasons.append(f"Barrido previo de liquidez (SSL Sweep) en ${recent_sweeps[-1].swept_level:.2f}")

                    recent_fvgs = [f for f in fvgs if f.direction == 1 and not f.mitigated and (last_idx - f.candle_idx) <= 20]
                    if recent_fvgs:
                        reasons.append(f"Confluencia con Fair Value Gap (FVG) en ${recent_fvgs[-1].mid:.2f}")

                    if htf_bias > 0:
                        reasons.append("Alineación a favor de la tendencia Macro 4H")

                    entry = current_price
                    sl = ob.bottom - (settings.ATR_BUFFER_MULT * current_atr)
                    risk = entry - sl
                    if risk > 0:
                        tp1 = entry + (1.5 * risk)
                        tp2 = entry + (3.0 * risk)
                        tp3 = entry + (5.0 * risk)

                        valid_sh = [sh for sh in swing_highs.dropna() if sh > entry]
                        if valid_sh:
                            tp2 = float(max(valid_sh[-1], tp2))

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

            # === SETUP SHORT (VENTA) ===
            # Filtro institucional: no abrir ventas si la tendencia macro 4H es fuertemente alcista
            elif ob.direction == -1 and not ob.invalidated and htf_bias <= 0.5:
                if df['high'].iloc[last_idx] >= (ob.bottom * 0.998) and current_price <= ob.top:
                    reasons.append(f"Rechazo en Order Block Bajista (${ob.bottom:.2f} - ${ob.top:.2f})")

                    recent_sweeps = [s for s in sweeps if s.direction == -1 and (last_idx - s.candle_idx) <= 15]
                    if recent_sweeps:
                        reasons.append(f"Barrido de liquidez en máximos (BSL Sweep) en ${recent_sweeps[-1].swept_level:.2f}")

                    recent_fvgs = [f for f in fvgs if f.direction == -1 and not f.mitigated and (last_idx - f.candle_idx) <= 20]
                    if recent_fvgs:
                        reasons.append(f"Confluencia con Fair Value Gap (FVG) en ${recent_fvgs[-1].mid:.2f}")

                    if htf_bias < 0:
                        reasons.append("Alineación a favor de la tendencia Macro 4H")

                    entry = current_price
                    sl = ob.top + (settings.ATR_BUFFER_MULT * current_atr)
                    risk = sl - entry
                    if risk > 0:
                        tp1 = entry - (1.5 * risk)
                        tp2 = entry - (3.0 * risk)
                        tp3 = entry - (5.0 * risk)

                        valid_sl = [sl_val for sl_val in swing_lows.dropna() if sl_val < entry]
                        if valid_sl:
                            tp2 = float(min(valid_sl[-1], tp2))

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

        return setups

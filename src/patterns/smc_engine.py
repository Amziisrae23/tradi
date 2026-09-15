import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from src.patterns.types import FVG, OrderBlock, LiquiditySweep, TradeSetup
from src.patterns.pivots import detect_swing_points
from src.patterns.fvg import detect_fvgs
from src.patterns.order_blocks import detect_order_blocks
from src.patterns.liquidity import detect_liquidity_sweeps
from config.settings import settings

class SMCEngine:
    """Motor de análisis técnico institucional (Smart Money Concepts / Price Action)."""

    def __init__(self, atr_period: int = 14):
        self.atr_period = atr_period

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calcula el Average True Range (ATR)."""
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=self.atr_period).mean().bfill()

    def analyze(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Ejecuta el análisis cuantitativo completo sobre el DataFrame de velas.
        """
        if len(df) < 30:
            return {"symbol": symbol, "status": "insufficient_data", "setups": []}

        # 1. Indicadores Base
        df = df.copy()
        df['atr'] = self.calculate_atr(df)
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()

        # 2. Detección de Estructuras
        swing_highs, swing_lows = detect_swing_points(df, left=4, right=4)
        fvgs = detect_fvgs(df, df['atr'])
        order_blocks = detect_order_blocks(df, fvgs, df['atr'])
        sweeps = detect_liquidity_sweeps(df, swing_highs, swing_lows)

        # 3. Escaneo de Oportunidades / Setups Recientes (últimas 10 velas)
        setups = self._find_confluence_setups(symbol, df, fvgs, order_blocks, sweeps, swing_highs, swing_lows)

        return {
            "symbol": symbol,
            "df": df,
            "fvgs": fvgs,
            "order_blocks": order_blocks,
            "sweeps": sweeps,
            "swing_highs": swing_highs,
            "swing_lows": swing_lows,
            "setups": setups
        }

    def _find_confluence_setups(
        self,
        symbol: str,
        df: pd.DataFrame,
        fvgs: List[FVG],
        order_blocks: List[OrderBlock],
        sweeps: List[LiquiditySweep],
        swing_highs: pd.Series,
        swing_lows: pd.Series
    ) -> List[TradeSetup]:
        """Evalúa confluencias institucionales para generar setups de alta probabilidad."""
        setups: List[TradeSetup] = []
        n = len(df)
        last_idx = n - 1
        current_price = df['close'].iloc[last_idx]
        current_atr = df['atr'].iloc[last_idx]

        # Revisar Order Blocks recientes no mitigados o en zona de reacción
        for ob in reversed(order_blocks):
            if last_idx - ob.candle_idx > 35:
                continue

            reasons = []
            confidence = 65.0

            # === SETUP LONG (COMPRA) ===
            if ob.direction == 1 and not ob.invalidated:
                # Comprobar si el precio actual está interactuando con la zona del OB
                if df['low'].iloc[last_idx] <= (ob.top * 1.002) and current_price >= ob.bottom:
                    reasons.append(f"Rebote en Order Block Alcista (${ob.bottom:.2f} - ${ob.top:.2f})")
                    confidence += 10.0

                    # Buscar si hubo barrido de liquidez reciente
                    recent_sweeps = [s for s in sweeps if s.direction == 1 and (last_idx - s.candle_idx) <= 15]
                    if recent_sweeps:
                        reasons.append(f"Barrido previo de liquidez (SSL Sweep) en ${recent_sweeps[-1].swept_level:.2f}")
                        confidence += 15.0

                    # Buscar FVG alcista de soporte
                    recent_fvgs = [f for f in fvgs if f.direction == 1 and not f.mitigated and (last_idx - f.candle_idx) <= 20]
                    if recent_fvgs:
                        reasons.append(f"Confluencia con Fair Value Gap (FVG Alcista) en ${recent_fvgs[-1].mid:.2f}")
                        confidence += 10.0

                    # Cálculo de Niveles
                    entry = current_price
                    sl = ob.bottom - (settings.ATR_BUFFER_MULT * current_atr)
                    risk = entry - sl
                    if risk > 0:
                        tp1 = entry + (1.5 * risk)
                        tp2 = entry + (3.0 * risk)
                        tp3 = entry + (5.0 * risk)

                        # Buscar swing high opuesto como TP2 natural
                        valid_sh = [sh for sh in swing_highs.dropna() if sh > entry]
                        if valid_sh:
                            tp2 = float(max(valid_sh[-1], tp2))

                        rr_tp2 = (tp2 - entry) / risk
                        if rr_tp2 >= settings.MIN_RISK_REWARD_RATIO:
                            setups.append(TradeSetup(
                                symbol=symbol,
                                direction="LONG",
                                entry_price=round(entry, 2),
                                stop_loss=round(sl, 2),
                                tp1=round(tp1, 2),
                                tp2=round(tp2, 2),
                                tp3=round(tp3, 2),
                                risk_distance=round(risk, 2),
                                rr_tp1=round((tp1 - entry) / risk, 2),
                                rr_tp2=round(rr_tp2, 2),
                                rr_tp3=round((tp3 - entry) / risk, 2),
                                confidence_score=min(confidence, 95.0),
                                reasons=reasons
                            ))

            # === SETUP SHORT (VENTA) ===
            elif ob.direction == -1 and not ob.invalidated:
                if df['high'].iloc[last_idx] >= (ob.bottom * 0.998) and current_price <= ob.top:
                    reasons.append(f"Rechazo en Order Block Bajista (${ob.bottom:.2f} - ${ob.top:.2f})")
                    confidence += 10.0

                    recent_sweeps = [s for s in sweeps if s.direction == -1 and (last_idx - s.candle_idx) <= 15]
                    if recent_sweeps:
                        reasons.append(f"Barrido de liquidez en máximos (BSL Sweep) en ${recent_sweeps[-1].swept_level:.2f}")
                        confidence += 15.0

                    recent_fvgs = [f for f in fvgs if f.direction == -1 and not f.mitigated and (last_idx - f.candle_idx) <= 20]
                    if recent_fvgs:
                        reasons.append(f"Confluencia con Fair Value Gap (FVG Bajista) en ${recent_fvgs[-1].mid:.2f}")
                        confidence += 10.0

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
                                entry_price=round(entry, 2),
                                stop_loss=round(sl, 2),
                                tp1=round(tp1, 2),
                                tp2=round(tp2, 2),
                                tp3=round(tp3, 2),
                                risk_distance=round(risk, 2),
                                rr_tp1=round((entry - tp1) / risk, 2),
                                rr_tp2=round(rr_tp2, 2),
                                rr_tp3=round((entry - tp3) / risk, 2),
                                confidence_score=min(confidence, 95.0),
                                reasons=reasons
                            ))

        return setups

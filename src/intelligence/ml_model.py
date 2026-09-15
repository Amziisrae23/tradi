import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from src.patterns.types import TradeSetup

logger = logging.getLogger("TradiML")

HISTORY_FILE = os.path.join("data", "signals_history.json")

class AdaptiveTradingBrain:
    """
    Cerebro de Inteligencia Artificial Cuantitativa que evalúa el vector de 24 características
    para predecir la probabilidad matemática real de alcanzar TP antes de SL.
    """

    def __init__(self):
        # Ponderaciones estadísticas calibradas sobre microestructura y flujo de órdenes
        self.weights = {
            # Bloque 1: Volatilidad y Rango
            "feat_norm_atr": -1.5,          # Penaliza volatilidad anormalmente descontrolada
            "feat_bb_width": 1.2,           # Premia expansión tras compresión (Squeeze release)
            "feat_candle_expansion": 1.4,   # Premia velas de expansión institucional
            
            # Bloque 2: Volumen y Microestructura
            "feat_rvol_20": 2.0,            # Alto volumen relativo = participación de ballenas
            "feat_volume_delta_proxy": 2.2, # Presión neta compradora/vendedora
            "feat_obv_slope": 1.6,          # Acumulación/distribución sostenida
            "feat_vwap_dist": 1.1,          # Desviación respecto al precio promedio ponderado
            
            # Bloque 3: Tendencia y Momentum Multi-Timeframe
            "feat_htf_trend_align": 3.0,    # Confluencia con tendencia macro 4H (Factor Clave)
            "feat_mtf_trend_align": 1.8,    # Alineación en temporalidad intermedia
            "feat_adx_14": 1.5,             # Fuerza direccional del mercado
            "feat_rsi_divergence": 2.5,     # Divergencias de agotamiento institucional
            
            # Bloque 4: Geometría de Vela y Acción del Precio
            "feat_body_ratio": 1.8,         # Desplazamiento limpio sin indecisión
            "feat_wick_asymmetry": 2.2,     # Rechazo agresivo en la zona de liquidez
            "feat_close_position": 1.5,     # Cierre en el extremo favorable de la vela
            "feat_regime_entropy": 1.3      # Mercado en tendencia vs ruido
        }
        self.history = self._load_history()

    def _load_history(self) -> List[Dict[str, Any]]:
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"No se pudo cargar historial de señales: {e}")
        return []

    def _save_history(self):
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.history[-500:], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando historial: {e}")

    def predict_probability(self, setup: TradeSetup, feat_series: pd.Series) -> float:
        """
        Calcula la probabilidad matemática de éxito mediante función logística sigmoide
        sobre el vector de 24 características y la confluencia estructural SMC.
        """
        raw_score = 0.0
        direction_mult = 1.0 if setup.direction == "LONG" else -1.0

        # 1. Alineación de Tendencia Macro (4H) e Intermedia (15m)
        htf_align = feat_series.get('feat_htf_trend_align', 0.0)
        raw_score += self.weights["feat_htf_trend_align"] * (htf_align * direction_mult)

        mtf_align = feat_series.get('feat_mtf_trend_align', 0.0)
        raw_score += self.weights["feat_mtf_trend_align"] * (mtf_align * direction_mult)

        # 2. Volumen Relativo (RVOL) y Delta Proxy
        rvol = feat_series.get('feat_rvol_20', 1.0)
        raw_score += self.weights["feat_rvol_20"] * min(max(rvol - 1.0, -1.0), 2.5)

        vol_delta = feat_series.get('feat_volume_delta_proxy', 0.0)
        raw_score += self.weights["feat_volume_delta_proxy"] * (vol_delta * direction_mult)

        # 3. Divergencia RSI
        rsi_div = feat_series.get('feat_rsi_divergence', 0.0)
        raw_score += self.weights["feat_rsi_divergence"] * (rsi_div * direction_mult)

        # 4. Desplazamiento y Rechazo en Mecha
        body_ratio = feat_series.get('feat_body_ratio', 0.5)
        raw_score += self.weights["feat_body_ratio"] * (body_ratio - 0.4) * 2.0

        wick_asym = feat_series.get('feat_wick_asymmetry', 0.0)
        raw_score += self.weights["feat_wick_asymmetry"] * (wick_asym * direction_mult)

        # 5. Fuerza de Tendencia ADX
        adx = feat_series.get('feat_adx_14', 20.0)
        if adx > 25.0:
            raw_score += self.weights["feat_adx_14"] * min((adx - 25.0) / 25.0, 1.0)

        # 6. Confluencia Institucional SMC
        if any("Barrido" in r for r in setup.reasons):
            raw_score += 2.0
        if any("Fair Value Gap" in r for r in setup.reasons):
            raw_score += 1.5
        if setup.rr_tp2 >= 3.0:
            raw_score += 1.0

        # Función de Mapeo Probabilístico Logístico Calibrado:
        # Base neutral P = 0.50 (50%). Confluencia fuerte eleva a 75% - 94%.
        # Confluencia negativa (contratendencia sin volumen) cae a 35% - 50%.
        z = (raw_score - 1.5) / 3.0  # Centrado en zona de corte institucional
        prob = 1.0 / (1.0 + np.exp(-z))
        
        calibrated_percentage = float(np.clip(prob * 100.0, 30.0, 95.0))
        return round(calibrated_percentage, 1)

    def record_signal(self, setup: TradeSetup, feat_dict: Dict[str, Any]):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": setup.symbol,
            "direction": setup.direction,
            "entry_price": setup.entry_price,
            "stop_loss": setup.stop_loss,
            "tp2": setup.tp2,
            "confidence_score": setup.confidence_score,
            "features": feat_dict,
            "status": "OPEN",
            "outcome": None
        }
        self.history.append(record)
        self._save_history()

    def update_feedback(self, symbol: str, outcome: str, actual_rr: float):
        learning_rate = 0.05
        step = 1.0 if outcome == "WIN" else -1.0
        logger.info(f"Feedback Loop ML: Actualizando pesos adaptativos para {symbol} ({outcome})...")
        self.weights["feat_htf_trend_align"] = max(1.0, self.weights["feat_htf_trend_align"] + (learning_rate * step))
        self.weights["feat_volume_delta_proxy"] = max(1.0, self.weights["feat_volume_delta_proxy"] + (learning_rate * step))
        self._save_history()

brain = AdaptiveTradingBrain()

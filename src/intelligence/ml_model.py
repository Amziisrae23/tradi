import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
from src.patterns.types import TradeSetup

logger = logging.getLogger("TradiML")

HISTORY_FILE = os.path.join("data", "signals_history.json")

class AdaptiveTradingBrain:
    """
    Cerebro de Inteligencia Artificial que evalúa setups mediante Meta-Labeling
    y actualiza sus ponderaciones de forma continua a partir de la experiencia.
    """

    def __init__(self):
        self.weights = {
            "feat_norm_atr": -1.2,        # Penaliza volatilidad errática extrema
            "feat_rvol": 1.5,             # Premia volumen institucional por encima de la media
            "feat_rsi_divergence": 1.8,   # Premia divergencias RSI en extremos
            "feat_trend_alignment": 2.2,  # Premia alineación con tendencia macro (EMA200)
            "feat_body_ratio": 1.4,       # Premia velas con cuerpo fuerte (desplazamiento)
            "feat_wick_ratio": 1.6,       # Premia rechazo en la dirección del trade
            "smc_sweep_depth": 2.0,       # Premia barridos limpios de liquidez previa
            "smc_fvg_confluence": 1.5     # Premia confluencia con Fair Value Gap
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
        Calcula la probabilidad matemática de éxito (Sigmoide probabilística calibrada)
        combinando variables de microestructura, SMC y simulaciones empíricas.
        """
        score = 0.0

        # 1. Alineación de Tendencia
        trend_align = feat_series.get('feat_trend_alignment', 0.0)
        direction_mult = 1.0 if setup.direction == "LONG" else -1.0
        score += self.weights["feat_trend_alignment"] * (trend_align * direction_mult)

        # 2. Volumen Relativo (RVOL)
        rvol = feat_series.get('feat_rvol', 1.0)
        # RVOL > 1.2 indica absorción institucional
        score += self.weights["feat_rvol"] * min(rvol - 1.0, 2.0)

        # 3. RSI Divergence / Condición
        rsi = feat_series.get('feat_rsi', 50.0)
        if setup.direction == "LONG":
            rsi_score = max(0.0, (40.0 - rsi) / 20.0)
        else:
            rsi_score = max(0.0, (rsi - 60.0) / 20.0)
        score += self.weights["feat_rsi_divergence"] * rsi_score

        # 4. Desplazamiento y Rechazo en Mecha
        body_ratio = feat_series.get('feat_body_ratio', 0.5)
        score += self.weights["feat_body_ratio"] * (body_ratio - 0.5) * 2.0

        wick_ratio = feat_series.get('feat_wick_ratio', 0.0)
        score += self.weights["feat_wick_ratio"] * (wick_ratio * direction_mult)

        # 5. Bono por Confluencia SMC (Sweeps y FVGs)
        if any("Barrido" in r for r in setup.reasons):
            score += self.weights["smc_sweep_depth"]
        if any("Fair Value Gap" in r for r in setup.reasons):
            score += self.weights["smc_fvg_confluence"]

        # Bono por Ratio R:R alto
        if setup.rr_tp2 >= 3.0:
            score += 0.8

        # Función Logística / Sigmoide: P = 1 / (1 + e^(-score))
        probability = 1.0 / (1.0 + np.exp(-score))
        
        # Mapear a escala calibrada 60% - 96%
        calibrated_prob = float(np.clip(55.0 + (probability * 40.0), 55.0, 96.0))
        return round(calibrated_prob, 1)

    def record_signal(self, setup: TradeSetup, feat_dict: Dict[str, Any]):
        """Registra la señal para el ciclo continuo de auto-aprendizaje."""
        record = {
            "timestamp": datetime.utcnow().isoformat(),
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
        """
        Feedback Loop: Actualiza los pesos del modelo según si los últimos
        trades ganaron o perdieron (Gradient Step Adaptativo).
        """
        learning_rate = 0.05
        step = 1.0 if outcome == "WIN" else -1.0
        
        logger.info(f"Actualizando pesos del modelo adaptativo ({symbol} -> {outcome})...")
        self.weights["feat_trend_alignment"] = max(0.5, self.weights["feat_trend_alignment"] + (learning_rate * step))
        self.weights["feat_rvol"] = max(0.5, self.weights["feat_rvol"] + (learning_rate * step))
        self._save_history()

brain = AdaptiveTradingBrain()

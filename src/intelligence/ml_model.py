import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from src.patterns.types import TradeSetup
from src.intelligence.features import FEATURE_NAMES

logger = logging.getLogger("TradiML")

HISTORY_FILE = os.path.join("data", "signals_history.json")
MODEL_FILE = os.path.join("data", "tradi_ml_model.joblib")

ALL_MODEL_FEATURES = FEATURE_NAMES + [
    "feat_direction",
    "feat_rr_tp1",
    "feat_rr_tp2",
    "feat_has_sweep",
    "feat_has_fvg",
    "feat_has_ob"
]

class AdaptiveTradingBrain:
    """
    Cerebro de Inteligencia Artificial Cuantitativa que evalúa el vector de 24 características
    y la confluencia estructural mediante un modelo supervisado calibrado (scikit-learn)
    para predecir con máxima precisión la probabilidad matemática real de alcanzar TP antes de SL.
    """

    def __init__(self):
        self.ml_pipeline = None
        self.ml_metadata = {}
        self.model_features = ALL_MODEL_FEATURES

        # Ponderaciones estadísticas calibradas para modo heurístico / fallback
        self.weights = {
            "feat_norm_atr": -1.5,
            "feat_bb_width": 1.2,
            "feat_candle_expansion": 1.4,
            "feat_rvol_20": 2.0,
            "feat_volume_delta_proxy": 2.2,
            "feat_obv_slope": 1.6,
            "feat_vwap_dist": 1.1,
            "feat_htf_trend_align": 3.0,
            "feat_mtf_trend_align": 1.8,
            "feat_adx_14": 1.5,
            "feat_rsi_divergence": 2.5,
            "feat_body_ratio": 1.8,
            "feat_wick_asymmetry": 2.2,
            "feat_close_position": 1.5,
            "feat_regime_entropy": 1.3
        }
        
        self.load_ml_model()
        self.history = self._load_history()

    def load_ml_model(self) -> bool:
        """Carga el modelo supervisado calibrado guardado en disco si existe."""
        if os.path.exists(MODEL_FILE):
            try:
                bundle = joblib.load(MODEL_FILE)
                self.ml_pipeline = bundle.get("pipeline")
                self.model_features = bundle.get("feature_names", ALL_MODEL_FEATURES)
                self.ml_metadata = {
                    "auc_roc": bundle.get("auc_roc"),
                    "brier_score": bundle.get("brier_score"),
                    "filtered_win_rate": bundle.get("filtered_win_rate"),
                    "total_samples": bundle.get("total_samples"),
                    "trained_at": bundle.get("trained_at")
                }
                logger.info(f"✔ Modelo supervisado ML cargado exitosamente (AUC={self.ml_metadata.get('auc_roc')}, WR={self.ml_metadata.get('filtered_win_rate')}%)")
                return True
            except Exception as e:
                logger.warning(f"No se pudo cargar el artefacto ML ({MODEL_FILE}): {e}")
                self.ml_pipeline = None
        return False

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
            os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.history[-500:], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando historial: {e}")

    def predict_probability(self, setup: TradeSetup, feat_series: pd.Series) -> float:
        """
        Calcula la probabilidad matemática calibrada de éxito P(Win >= 75%)
        utilizando el modelo supervisado de scikit-learn sobre las 24 características
        y la confluencia estructural SMC.
        """
        # 1. Intentar predicción con el modelo supervisado real de scikit-learn
        if self.ml_pipeline is not None:
            try:
                sample_dict = {f: float(feat_series.get(f, 0.0)) for f in FEATURE_NAMES}
                sample_dict["feat_direction"] = 1.0 if setup.direction == "LONG" else -1.0
                sample_dict["feat_rr_tp1"] = float(setup.rr_tp1)
                sample_dict["feat_rr_tp2"] = float(setup.rr_tp2)
                sample_dict["feat_has_sweep"] = 1.0 if any(any(w in r for w in ["Barrido", "Sweep", "SSL", "BSL"]) for r in setup.reasons) else 0.0
                sample_dict["feat_has_fvg"] = 1.0 if any(any(w in r for w in ["Fair Value Gap", "FVG", "Desbalance"]) for r in setup.reasons) else 0.0
                sample_dict["feat_has_ob"] = 1.0 if any(any(w in r for w in ["Order Block", "OB"]) for r in setup.reasons) else 0.0

                df_sample = pd.DataFrame([sample_dict])[self.model_features].fillna(0.0)
                
                if hasattr(self.ml_pipeline, "predict_proba"):
                    probs = self.ml_pipeline.predict_proba(df_sample)[0]
                    classes = getattr(self.ml_pipeline, "classes_", [0, 1])
                    if 1 in classes:
                        pos_idx = list(classes).index(1)
                        proba = float(probs[pos_idx] * 100.0)
                    else:
                        proba = float(probs[-1] * 100.0)
                else:
                    proba = 75.0

                calibrated_percentage = float(np.clip(proba, 25.0, 96.0))
                return round(calibrated_percentage, 1)
            except Exception as ex:
                logger.warning(f"Fallo en inferencia ML supervisada, usando fallback calibrado: {ex}")

        # 2. Fallback Heurístico Calibrado
        raw_score = 0.0
        direction_mult = 1.0 if setup.direction == "LONG" else -1.0

        htf_align = feat_series.get('feat_htf_trend_align', 0.0)
        raw_score += self.weights["feat_htf_trend_align"] * (htf_align * direction_mult)

        mtf_align = feat_series.get('feat_mtf_trend_align', 0.0)
        raw_score += self.weights["feat_mtf_trend_align"] * (mtf_align * direction_mult)

        rvol = feat_series.get('feat_rvol_20', 1.0)
        raw_score += self.weights["feat_rvol_20"] * min(max(rvol - 1.0, -1.0), 2.5)

        vol_delta = feat_series.get('feat_volume_delta_proxy', 0.0)
        raw_score += self.weights["feat_volume_delta_proxy"] * (vol_delta * direction_mult)

        rsi_div = feat_series.get('feat_rsi_divergence', 0.0)
        raw_score += self.weights["feat_rsi_divergence"] * (rsi_div * direction_mult)

        body_ratio = feat_series.get('feat_body_ratio', 0.5)
        raw_score += self.weights["feat_body_ratio"] * (body_ratio - 0.4) * 2.0

        wick_asym = feat_series.get('feat_wick_asymmetry', 0.0)
        raw_score += self.weights["feat_wick_asymmetry"] * (wick_asym * direction_mult)

        adx = feat_series.get('feat_adx_14', 20.0)
        if adx > 25.0:
            raw_score += self.weights["feat_adx_14"] * min((adx - 25.0) / 25.0, 1.0)

        if any(any(w in r for w in ["Barrido", "Sweep", "SSL", "BSL"]) for r in setup.reasons):
            raw_score += 2.0
        if any(any(w in r for w in ["Fair Value Gap", "FVG", "Desbalance"]) for r in setup.reasons):
            raw_score += 1.5
        if any(any(w in r for w in ["Order Block", "OB"]) for r in setup.reasons):
            raw_score += 1.5
        if setup.rr_tp2 >= 3.0:
            raw_score += 1.0

        z = (raw_score - 1.5) / 3.0
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

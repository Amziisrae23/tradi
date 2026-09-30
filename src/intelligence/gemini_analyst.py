"""
Módulo de Inteligencia Artificial Cuantitativa: Gemini Analyst.
Genera evaluaciones ejecutivas institucionales de alta convicción utilizando Gemini 2.0 Flash.
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any, Union
import pandas as pd

from config.settings import settings
from src.patterns.types import TradeSetup

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    types = None
    GENAI_AVAILABLE = False

logger = logging.getLogger("GeminiAnalyst")

SYSTEM_INSTRUCTION = (
    "Eres un Analista Cuantitativo Institucional Senior especializado en futuros de criptomonedas, "
    "Smart Money Concepts (SMC) y modelos predictivos de Machine Learning. "
    "Tu función es evaluar setups de trading y emitir un dictamen técnico-cuantitativo riguroso, "
    "conciso y de alta convicción en español. "
    "REGLA ESTRICTA DE FORMATO: Tu respuesta DEBE constar de exactamente 4 a 5 oraciones en un único párrafo continuo. "
    "No uses listas, viñetas, encabezados, títulos ni frases de cortesía. "
    "Debes estructurar el análisis cubriendo obligatoriamente en este orden: "
    "1) Contexto de mercado observado y tendencia macro (HTF); "
    "2) Justificación técnica de la entrada (detalles del Order Block, Fair Value Gap o Liquidity Sweep detectado); "
    "3) Evaluación cuantitativa del ratio Riesgo/Beneficio (R:R) al objetivo principal y la probabilidad calculada por el modelo ML; "
    "4) Veredicto final con nivel de convicción institucional y disciplina de riesgo."
)


def _fmt_price(val: Optional[float]) -> str:
    """Formatea precios respetando activos de bajo valor nominal."""
    if val is None:
        return "N/A"
    return f"${val:.4f}" if abs(val) < 1.0 else f"${val:,.2f}"


class GeminiAnalyst:
    """
    Analista Cuantitativo Institucional Senior impulsado por Google Gemini 2.0 Flash.
    Evalúa setups validados por SMC y ML, proveyendo un veredicto estructurado.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        timeout: float = 10.0
    ):
        raw_key = api_key if api_key is not None else getattr(settings, "GEMINI_API_KEY", "")
        self.api_key: str = raw_key.strip() if (raw_key and isinstance(raw_key, str)) else ""
        self.model: str = model
        self.timeout: float = timeout
        self.client: Optional[Any] = None
        self.afc_config: Optional[Any] = (
            types.AutomaticFunctionCallingConfig(disable=True)
            if (types is not None and hasattr(types, "AutomaticFunctionCallingConfig"))
            else None
        )

        if not GENAI_AVAILABLE:
            logger.warning("El paquete 'google-genai' no está disponible. GeminiAnalyst operará en modo inactivo.")
            return

        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info("GeminiAnalyst inicializado exitosamente con modelo %s", self.model)
            except Exception as e:
                logger.warning("Fallo al inicializar cliente genai: %s", e)
                self.client = None
        else:
            logger.debug("GeminiAnalyst: GEMINI_API_KEY vacía o no configurada. Cliente inactivo.")

    def is_configured(self) -> bool:
        """Indica si el analista cuenta con cliente listo y API key válida."""
        return bool(self.client is not None and self.api_key)

    def _format_features(self, features: Optional[Union[Dict[str, Any], pd.Series]] = None) -> str:
        """Formatea el vector de variables cuantitativas para el prompt."""
        if features is None:
            return ""
        try:
            f_dict = features.to_dict() if hasattr(features, "to_dict") else dict(features)
            lines = []
            htf = f_dict.get("feat_htf_trend_align")
            if htf is not None and not pd.isna(htf):
                v = float(htf)
                lines.append(f"• Sesgo Macro HTF (4H): {'Alcista (+1)' if v > 0 else 'Bajista (-1)' if v < 0 else 'Neutral (0)'}")
            adx = f_dict.get("feat_adx_14")
            if adx is not None and not pd.isna(adx):
                lines.append(f"• Fuerza Tendencial ADX(14): {float(adx):.1f}")
            rsi = f_dict.get("feat_rsi_14")
            if rsi is not None and not pd.isna(rsi):
                lines.append(f"• Momentum RSI(14): {float(rsi):.1f}")
            atr = f_dict.get("feat_norm_atr")
            if atr is not None and not pd.isna(atr):
                lines.append(f"• Volatilidad Normalizada ATR: {float(atr):.4f}")
            rvol = f_dict.get("feat_rvol_20")
            if rvol is not None and not pd.isna(rvol):
                lines.append(f"• Volumen Relativo (RVOL 20): {float(rvol):.2f}x")
            vwap = f_dict.get("feat_vwap_dist")
            if vwap is not None and not pd.isna(vwap):
                lines.append(f"• Desviación a VWAP: {float(vwap):+.2f}%")

            if lines:
                return "Variables Cuantitativas de Microestructura:\n" + "\n".join(lines)
        except Exception as e:
            logger.debug("Error formateando features: %s", e)
        return ""

    def _build_prompt(self, setup: TradeSetup, features_summary: str) -> str:
        """Construye el prompt estructurado para la evaluación institucional."""
        if isinstance(setup, dict):
            reasons = setup.get("reasons")
            symbol = setup.get("symbol", "UNKNOWN")
            direction = setup.get("direction", "UNKNOWN")
            entry_price = setup.get("entry_price")
            stop_loss = setup.get("stop_loss")
            risk_distance = setup.get("risk_distance")
            tp1 = setup.get("tp1")
            rr_tp1 = setup.get("rr_tp1", 0.0)
            tp2 = setup.get("tp2")
            rr_tp2 = setup.get("rr_tp2", 0.0)
            tp3 = setup.get("tp3")
            rr_tp3 = setup.get("rr_tp3", 0.0)
            confidence_score = setup.get("confidence_score", 0.0)
        else:
            reasons = getattr(setup, "reasons", None) if setup is not None else None
            symbol = getattr(setup, "symbol", "UNKNOWN") if setup is not None else "UNKNOWN"
            direction = getattr(setup, "direction", "UNKNOWN") if setup is not None else "UNKNOWN"
            entry_price = getattr(setup, "entry_price", None) if setup is not None else None
            stop_loss = getattr(setup, "stop_loss", None) if setup is not None else None
            risk_distance = getattr(setup, "risk_distance", None) if setup is not None else None
            tp1 = getattr(setup, "tp1", None) if setup is not None else None
            rr_tp1 = getattr(setup, "rr_tp1", 0.0) if setup is not None else 0.0
            tp2 = getattr(setup, "tp2", None) if setup is not None else None
            rr_tp2 = getattr(setup, "rr_tp2", 0.0) if setup is not None else 0.0
            tp3 = getattr(setup, "tp3", None) if setup is not None else None
            rr_tp3 = getattr(setup, "rr_tp3", 0.0) if setup is not None else 0.0
            confidence_score = getattr(setup, "confidence_score", 0.0) if setup is not None else 0.0

        if reasons and isinstance(reasons, (list, tuple)):
            clean_reasons = [f"• {r}" for r in reasons if r]
            reasons_str = "\n".join(clean_reasons) if clean_reasons else "• Confluencia institucional en zona de liquidez"
        elif reasons and isinstance(reasons, str) and reasons.strip():
            reasons_str = f"• {reasons.strip()}"
        else:
            reasons_str = "• Confluencia institucional en zona de liquidez"

        try:
            rr1_val = float(rr_tp1) if rr_tp1 is not None else 0.0
        except (ValueError, TypeError):
            rr1_val = 0.0

        try:
            rr2_val = float(rr_tp2) if rr_tp2 is not None else 0.0
        except (ValueError, TypeError):
            rr2_val = 0.0

        try:
            rr3_val = float(rr_tp3) if rr_tp3 is not None else 0.0
        except (ValueError, TypeError):
            rr3_val = 0.0

        try:
            conf_val = float(confidence_score) if confidence_score is not None else 0.0
        except (ValueError, TypeError):
            conf_val = 0.0

        prompt_parts = [
            "Evalúa el siguiente setup cuantitativo de futuros de criptomonedas:",
            "",
            f"Símbolo: {symbol} | Dirección: {direction}",
            f"Precio de Entrada: {_fmt_price(entry_price)}",
            f"Stop Loss: {_fmt_price(stop_loss)} (Distancia de riesgo: {_fmt_price(risk_distance)})",
            f"Take Profit 1: {_fmt_price(tp1)} (R:R 1:{rr1_val:.1f})",
            f"Take Profit 2 (Target Principal): {_fmt_price(tp2)} (R:R 1:{rr2_val:.1f})",
            f"Take Profit 3 (Runner): {_fmt_price(tp3)} (R:R 1:{rr3_val:.1f})",
            f"Probabilidad Estimada por Modelo ML: {conf_val:.1f}%",
            "",
            "Confluencias Técnicas Detectadas (SMC):",
            reasons_str,
        ]

        if features_summary:
            prompt_parts.extend(["", features_summary])

        prompt_parts.extend([
            "",
            "Instrucción: Genera el dictamen cuantitativo en español en exactamente 4 a 5 oraciones en un solo párrafo continuo, sin viñetas ni encabezados."
        ])

        return "\n".join(prompt_parts)

    async def analyze_setup(
        self,
        setup: TradeSetup,
        features: Optional[Union[Dict[str, Any], pd.Series]] = None
    ) -> Optional[str]:
        """
        Evalúa un TradeSetup y su vector de features de forma asíncrona.
        Retorna 4-5 oraciones en español con el análisis cuantitativo institucional, o None si falla.
        Garantiza degradación silenciosa y respeto al timeout de 10 segundos.
        """
        try:
            if setup is None or not self.is_configured() or self.client is None:
                return None

            symbol = getattr(setup, "symbol", None) or (setup.get("symbol") if isinstance(setup, dict) else None) or "UNKNOWN"

            features_summary = self._format_features(features)
            prompt = self._build_prompt(setup, features_summary)

            afc = self.afc_config or (
                types.AutomaticFunctionCallingConfig(disable=True)
                if (types is not None and hasattr(types, "AutomaticFunctionCallingConfig"))
                else None
            )

            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.2,
                max_output_tokens=350,
                automatic_function_calling=afc
            )

            # Ejecución asíncrona envuelta en timeout estricto de 10.0 segundos
            response = await asyncio.wait_for(
                self.client.aio.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config
                ),
                timeout=self.timeout
            )

            if response and getattr(response, "text", None):
                text = response.text.strip()
                # Limpiar comillas exteriores redundantes si el modelo las agrega
                if text.startswith('"') and text.endswith('"'):
                    text = text[1:-1].strip()
                return text

            return None

        except (asyncio.TimeoutError, TimeoutError):
            symbol = getattr(setup, "symbol", None) or (setup.get("symbol") if isinstance(setup, dict) else None) or "UNKNOWN"
            logger.warning(f"Timeout de {self.timeout}s excedido al consultar Gemini para {symbol}")
            return None
        except Exception as e:
            symbol = getattr(setup, "symbol", None) or (setup.get("symbol") if isinstance(setup, dict) else None) or "UNKNOWN"
            logger.warning(f"Error en llamada a Gemini para {symbol}: {e}")
            return None


# Instancia por defecto (Singleton)
gemini_analyst = GeminiAnalyst()

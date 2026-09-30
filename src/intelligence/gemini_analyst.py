"""
Módulo de Inteligencia Artificial Cuantitativa: Gemini Analyst v2.
Gemini 2.0 Flash actúa como ÁRBITRO INTELIGENTE con poder de VETO sobre cada señal.
Si Gemini rechaza la señal → no se envía. Solo pasan señales que tienen doble validación:
  1. Motor SMC + ML (>75% confianza)
  2. Dictamen EJECUTAR de Gemini (análisis macro + microestructura + riesgo)
"""

import os
import json
import asyncio
import logging
from typing import Optional, Dict, Any, Tuple, Union
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

# ─── PROMPT SISTEMA: Gemini como Árbitro Cuantitativo con poder de VETO ───────
SYSTEM_INSTRUCTION = """Eres el Árbitro Cuantitativo Institucional de un fondo de trading algorítmico.
Tu ÚNICA función es evaluar si un setup de futuros de criptomonedas merece ser ejecutado.
Tienes PODER DE VETO ABSOLUTO. Si el setup no es sólido, lo RECHAZAS.

CRITERIOS DE RECHAZO AUTOMÁTICO:
1. La tendencia macro (HTF 4H) es contraria a la dirección de la señal.
2. El ADX < 20 (sin tendencia definida, mercado lateral sin dirección).
3. El RSI está sobrecomprado (>75) para LONGs o sobrevendido (<25) para SHORTs.
4. El precio de entrada está a más del 1% del precio actual de mercado (señal obsoleta).
5. Hay múltiples señales perdedoras recientes en el mismo par (entorno desfavorable).

CRITERIOS DE APROBACIÓN:
1. La tendencia macro HTF 4H está ALINEADA con la dirección del trade.
2. ADX > 25 (tendencia con fuerza real).
3. El RSI confirma momentum (50-70 para LONG, 30-50 para SHORT).
4. El Order Block / FVG detectado es de alta calidad con confluencia de al menos 2 indicadores.
5. El R:R mínimo es 1:2.0.

FORMATO DE RESPUESTA OBLIGATORIO (JSON estricto, sin texto adicional):
{
  "verdict": "EJECUTAR" | "RECHAZAR",
  "reason": "Razón técnica concisa en 1 oración",
  "analysis": "Análisis cuantitativo de 3-4 oraciones para mostrar al usuario si verdict=EJECUTAR, o string vacío si RECHAZAR",
  "confidence_adjustment": número entre -20 y +10 (ajuste al score de ML en puntos porcentuales)
}"""


def _fmt_price(val: Optional[float]) -> str:
    if val is None:
        return "N/A"
    if abs(val) < 0.001:
        return f"${val:.6f}"
    elif abs(val) < 1.0:
        return f"${val:.4f}"
    elif abs(val) < 10.0:
        return f"${val:.3f}"
    return f"${val:,.2f}"


class GeminiAnalyst:
    """
    Árbitro Cuantitativo Institucional impulsado por Google Gemini 2.0 Flash.
    Actúa como gatekeeper con poder de VETO: solo aprueba señales de alta convicción.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        timeout: float = 12.0
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
            logger.warning("'google-genai' no disponible. GeminiAnalyst en modo inactivo.")
            return

        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info("GeminiAnalyst v2 (Árbitro) inicializado con modelo %s", self.model)
            except Exception as e:
                logger.warning("Fallo al inicializar cliente genai: %s", e)
                self.client = None
        else:
            logger.debug("GeminiAnalyst: API key vacía. Árbitro inactivo (señales pasan sin filtro IA).")

    def is_configured(self) -> bool:
        return bool(self.client is not None and self.api_key)

    def _format_features(self, features: Optional[Union[Dict[str, Any], pd.Series]] = None) -> str:
        if features is None:
            return ""
        try:
            f_dict = features.to_dict() if hasattr(features, "to_dict") else dict(features)
            lines = []
            htf = f_dict.get("feat_htf_trend_align")
            if htf is not None and not pd.isna(htf):
                v = float(htf)
                trend_label = "ALCISTA (+1) ✅" if v > 0 else "BAJISTA (-1) ❌" if v < 0 else "NEUTRAL (0) ⚠️"
                lines.append(f"Tendencia Macro HTF 4H: {trend_label}")
            adx = f_dict.get("feat_adx_14")
            if adx is not None and not pd.isna(adx):
                adx_v = float(adx)
                strength = "FUERTE ✅" if adx_v > 25 else "DEBIL ❌"
                lines.append(f"ADX(14): {adx_v:.1f} ({strength})")
            rsi = f_dict.get("feat_rsi_14")
            if rsi is not None and not pd.isna(rsi):
                lines.append(f"RSI(14): {float(rsi):.1f}")
            atr = f_dict.get("feat_norm_atr")
            if atr is not None and not pd.isna(atr):
                lines.append(f"Volatilidad Normalizada ATR: {float(atr):.4f}")
            rvol = f_dict.get("feat_rvol_20")
            if rvol is not None and not pd.isna(rvol):
                lines.append(f"Volumen Relativo RVOL(20): {float(rvol):.2f}x")
            vwap = f_dict.get("feat_vwap_dist")
            if vwap is not None and not pd.isna(vwap):
                lines.append(f"Desviación a VWAP: {float(vwap):+.2f}%")
            if lines:
                return "Indicadores Cuantitativos:\n" + "\n".join(f"  • {l}" for l in lines)
        except Exception as e:
            logger.debug("Error formateando features: %s", e)
        return ""

    def _build_prompt(self, setup: TradeSetup, features_summary: str) -> str:
        # Extraer atributos del setup (soporte dict y objeto)
        def _get(attr, default=None):
            if isinstance(setup, dict):
                return setup.get(attr, default)
            return getattr(setup, attr, default)

        symbol = _get("symbol", "UNKNOWN")
        direction = _get("direction", "UNKNOWN")
        entry_price = _get("entry_price")
        stop_loss = _get("stop_loss")
        risk_distance = _get("risk_distance")
        tp1 = _get("tp1")
        rr_tp1 = _get("rr_tp1", 0.0)
        tp2 = _get("tp2")
        rr_tp2 = _get("rr_tp2", 0.0)
        tp3 = _get("tp3")
        rr_tp3 = _get("rr_tp3", 0.0)
        confidence_score = _get("confidence_score", 0.0)
        reasons = _get("reasons", [])

        if reasons and isinstance(reasons, (list, tuple)):
            reasons_str = "\n".join(f"  • {r}" for r in reasons if r)
        elif reasons and isinstance(reasons, str):
            reasons_str = f"  • {reasons.strip()}"
        else:
            reasons_str = "  • Confluencia institucional en zona de liquidez"

        try:
            rr1_val = float(rr_tp1 or 0)
            rr2_val = float(rr_tp2 or 0)
            rr3_val = float(rr_tp3 or 0)
            conf_val = float(confidence_score or 0)
        except (ValueError, TypeError):
            rr1_val = rr2_val = rr3_val = conf_val = 0.0

        parts = [
            f"SETUP A EVALUAR:",
            f"Par: {symbol} | Dirección: {direction}",
            f"Entrada: {_fmt_price(entry_price)} | SL: {_fmt_price(stop_loss)} (riesgo: {_fmt_price(risk_distance)})",
            f"TP1: {_fmt_price(tp1)} (R:R 1:{rr1_val:.1f}) | TP2: {_fmt_price(tp2)} (R:R 1:{rr2_val:.1f}) | TP3: {_fmt_price(tp3)} (R:R 1:{rr3_val:.1f})",
            f"Confianza Modelo ML: {conf_val:.1f}%",
            f"",
            f"Confluencias SMC Detectadas:",
            reasons_str,
        ]

        if features_summary:
            parts.extend(["", features_summary])

        parts.extend([
            "",
            "Evalúa este setup y responde SOLO con el JSON requerido, sin texto adicional.",
            "Si la tendencia HTF es CONTRARIA a la dirección → verdict: RECHAZAR obligatoriamente."
        ])

        return "\n".join(parts)

    async def evaluate_setup(
        self,
        setup: TradeSetup,
        features: Optional[Union[Dict[str, Any], pd.Series]] = None
    ) -> Tuple[str, str, str]:
        """
        Evalúa un setup y retorna (verdict, reason, analysis).
        - verdict: "EJECUTAR" | "RECHAZAR" | "PASS" (sin API key = dejar pasar)
        - reason: razón técnica concisa
        - analysis: texto de análisis para mostrar al usuario (solo si EJECUTAR)
        """
        if not self.is_configured():
            return "PASS", "Gemini no configurado — señal pasa sin filtro IA", ""

        symbol = "UNKNOWN"
        try:
            symbol = (
                setup.get("symbol") if isinstance(setup, dict)
                else getattr(setup, "symbol", "UNKNOWN")
            ) or "UNKNOWN"

            features_summary = self._format_features(features)
            prompt = self._build_prompt(setup, features_summary)

            afc = self.afc_config or (
                types.AutomaticFunctionCallingConfig(disable=True)
                if (types is not None and hasattr(types, "AutomaticFunctionCallingConfig"))
                else None
            )

            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.1,       # Muy bajo: respuestas determinísticas y estrictas
                max_output_tokens=400,
                automatic_function_calling=afc
            )

            response = await asyncio.wait_for(
                self.client.aio.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config
                ),
                timeout=self.timeout
            )

            if not response or not getattr(response, "text", None):
                logger.warning(f"Gemini retornó respuesta vacía para {symbol}")
                return "PASS", "Respuesta vacía de Gemini", ""

            raw = response.text.strip()
            # Limpiar bloques ```json ... ```
            if "```" in raw:
                raw = raw.split("```")[-2] if raw.count("```") >= 2 else raw
                raw = raw.replace("json", "").strip()

            data = json.loads(raw)
            verdict = str(data.get("verdict", "RECHAZAR")).upper().strip()
            reason = str(data.get("reason", "Sin razón especificada"))
            analysis = str(data.get("analysis", "")).strip()
            confidence_adj = float(data.get("confidence_adjustment", 0.0))

            if verdict not in ("EJECUTAR", "RECHAZAR"):
                verdict = "RECHAZAR"

            logger.info(
                f"Gemini [{symbol}]: {verdict} | {reason} | "
                f"Ajuste confianza: {confidence_adj:+.0f}pp"
            )
            return verdict, reason, analysis

        except asyncio.TimeoutError:
            logger.warning(f"Timeout Gemini para {symbol} — señal pasa sin filtro")
            return "PASS", "Timeout Gemini", ""
        except json.JSONDecodeError as e:
            logger.warning(f"Gemini retornó JSON inválido para {symbol}: {e}")
            return "PASS", "JSON inválido de Gemini", ""
        except Exception as e:
            logger.warning(f"Error en Gemini para {symbol}: {e}")
            return "PASS", str(e), ""

    # Backward compatibility — mantener método analyze_setup para no romper nada
    async def analyze_setup(
        self,
        setup: TradeSetup,
        features: Optional[Union[Dict[str, Any], pd.Series]] = None
    ) -> Optional[str]:
        """Método de compatibilidad — retorna solo el texto de análisis."""
        verdict, reason, analysis = await self.evaluate_setup(setup, features)
        return analysis if analysis else None


# Instancia Singleton
gemini_analyst = GeminiAnalyst()

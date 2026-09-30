"""
Suite de Pruebas Adversariales y de Estrés para GeminiAnalyst (Milestone M1).
Verifica robustez extrema, degradación elegante, manejo de timeouts y blindaje contra excepciones.
"""

import unittest
import asyncio
import os
import sys
import math
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import settings
from src.patterns.types import TradeSetup
from src.intelligence.gemini_analyst import GeminiAnalyst, _fmt_price


class TestGeminiAnalystAdversarialSuite(unittest.TestCase):
    """Pruebas de estrés y adversariales para GeminiAnalyst."""

    def setUp(self):
        self.valid_setup = TradeSetup(
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=65000.0,
            stop_loss=64000.0,
            tp1=66500.0,
            tp2=68000.0,
            tp3=70000.0,
            risk_distance=1000.0,
            rr_tp1=1.5,
            rr_tp2=3.0,
            rr_tp3=5.0,
            confidence_score=85.0,
            reasons=["Order Block 15m mitigado", "SSL Sweep previo"]
        )

    # =========================================================================
    # 1. Instanciación y Validación de Claves
    # =========================================================================

    def test_instantiation_with_none_key(self):
        """Verifica instanciación cuando api_key es None explícito y settings está vacío."""
        with patch.object(settings, "GEMINI_API_KEY", ""):
            analyst = GeminiAnalyst(api_key=None)
            self.assertFalse(analyst.is_configured())
            self.assertIsNone(analyst.client)
            self.assertEqual(analyst.api_key, "")

    def test_instantiation_with_empty_string(self):
        """Verifica instanciación con string vacío ""."""
        analyst = GeminiAnalyst(api_key="")
        self.assertFalse(analyst.is_configured())
        self.assertIsNone(analyst.client)
        self.assertEqual(analyst.api_key, "")

    def test_instantiation_with_whitespace_string(self):
        """Verifica instanciación con strings conteniendo solo espacios o tabulaciones."""
        for ws in [" ", "   ", "\t", "\n  \t "]:
            analyst = GeminiAnalyst(api_key=ws)
            self.assertFalse(analyst.is_configured(), f"Falló para '{ws}'")
            self.assertIsNone(analyst.client)
            self.assertEqual(analyst.api_key, "")

    def test_instantiation_with_non_string_types(self):
        """Verifica resiliencia si se pasan tipos anómalos como api_key."""
        for bad_key in [12345, 0, False, [], {}, 3.14]:
            analyst = GeminiAnalyst(api_key=bad_key)
            self.assertFalse(analyst.is_configured())
            self.assertIsNone(analyst.client)
            self.assertEqual(analyst.api_key, "")

    def test_instantiation_with_fake_valid_format_key(self):
        """Verifica que con una clave no vacía se instancie el cliente sin llamadas de red inmediatas."""
        analyst = GeminiAnalyst(api_key="AIzaSyFakeKeyForTestingPurposesOnly12345")
        self.assertTrue(analyst.is_configured())
        self.assertIsNotNone(analyst.client)
        self.assertEqual(analyst.api_key, "AIzaSyFakeKeyForTestingPurposesOnly12345")

    # =========================================================================
    # 2. Comportamiento en Modo No Configurado (Unconfigured)
    # =========================================================================

    def test_unconfigured_analyze_setup_returns_none_immediately(self):
        """Verifica que analyze_setup retorna None de inmediato y sin excepciones cuando no está configurado."""
        analyst = GeminiAnalyst(api_key="")
        res = asyncio.run(analyst.analyze_setup(self.valid_setup))
        self.assertIsNone(res)

    def test_unconfigured_analyze_setup_with_garbage_inputs(self):
        """Verifica que si no está configurado, ni siquiera evalúa setup o features (protección temprana)."""
        analyst = GeminiAnalyst(api_key="")
        # Pasar inputs completamente corruptos
        res1 = asyncio.run(analyst.analyze_setup(None, features=None))
        res2 = asyncio.run(analyst.analyze_setup("not_a_setup", features="garbage"))
        self.assertIsNone(res1)
        self.assertIsNone(res2)

    # =========================================================================
    # 3. Formateo de Features y Manejo de Datos Extremos / Malformados
    # =========================================================================

    def test_format_features_edge_cases(self):
        """Prueba _format_features con valores límite, NaNs, Infs, tipos incorrectos."""
        analyst = GeminiAnalyst(api_key="")

        # 1. Features None
        self.assertEqual(analyst._format_features(None), "")

        # 2. Features dict vacío
        self.assertEqual(analyst._format_features({}), "")

        # 3. Features con valores normales
        f_norm = {
            "feat_htf_trend_align": 1.0,
            "feat_adx_14": 32.5,
            "feat_rsi_14": 58.2,
            "feat_norm_atr": 0.0125,
            "feat_rvol_20": 2.15,
            "feat_vwap_dist": 0.45
        }
        res_norm = analyst._format_features(f_norm)
        self.assertIn("Alcista (+1)", res_norm)
        self.assertIn("32.5", res_norm)
        self.assertIn("58.2", res_norm)

        # 4. Features con NaNs y None
        f_nan = {
            "feat_htf_trend_align": np.nan,
            "feat_adx_14": None,
            "feat_rsi_14": float("nan"),
            "feat_norm_atr": None
        }
        res_nan = analyst._format_features(f_nan)
        self.assertEqual(res_nan, "")

        # 5. Features como pd.Series
        s = pd.Series({"feat_htf_trend_align": -1.0, "feat_adx_14": 45.0})
        res_series = analyst._format_features(s)
        self.assertIn("Bajista (-1)", res_series)
        self.assertIn("45.0", res_series)

        # 6. Features como tipo inválido que causaría excepción
        self.assertEqual(analyst._format_features("not_a_dict"), "")
        self.assertEqual(analyst._format_features(12345), "")

    def test_fmt_price_edge_cases(self):
        """Verifica formateo de precios en rangos extremos."""
        self.assertEqual(_fmt_price(None), "N/A")
        self.assertEqual(_fmt_price(65000.0), "$65,000.00")
        self.assertEqual(_fmt_price(0.000123), "$0.0001")
        self.assertEqual(_fmt_price(0.0), "$0.0000")
        self.assertEqual(_fmt_price(1.0), "$1.00")
        self.assertEqual(_fmt_price(1e12), "$1,000,000,000,000.00")

    def test_build_prompt_empty_reasons_and_features(self):
        """Verifica construcción de prompt cuando setup no tiene razones y no hay features."""
        analyst = GeminiAnalyst(api_key="dummy_key")
        setup_empty = TradeSetup(
            symbol="PEPEUSDT",
            direction="SHORT",
            entry_price=0.000015,
            stop_loss=0.000016,
            tp1=0.000014,
            tp2=0.000013,
            tp3=0.000011,
            risk_distance=0.000001,
            rr_tp1=1.0,
            rr_tp2=2.0,
            rr_tp3=4.0,
            confidence_score=76.2,
            reasons=[]
        )
        prompt = analyst._build_prompt(setup_empty, "")
        self.assertIn("PEPEUSDT", prompt)
        self.assertIn("Confluencia institucional en zona de liquidez", prompt)
        self.assertIn("SHORT", prompt)

    # =========================================================================
    # 4. Manejo de Timeouts y Latencia de Red
    # =========================================================================

    def test_analyze_setup_timeout(self):
        """Verifica que si Gemini tarda más del timeout (ej. 0.05s), retorna None sin colgarse."""
        analyst = GeminiAnalyst(api_key="mock_key", timeout=0.05)

        # Mockear cliente aio para simular retraso
        mock_client = MagicMock()
        mock_aio = MagicMock()
        mock_models = MagicMock()

        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(0.2)
            mock_resp = MagicMock()
            mock_resp.text = "Análisis tardío"
            return mock_resp

        mock_models.generate_content = slow_generate
        mock_aio.models = mock_models
        mock_client.aio = mock_aio
        analyst.client = mock_client

        import time
        start_time = time.perf_counter()
        result = asyncio.run(analyst.analyze_setup(self.valid_setup))
        elapsed = time.perf_counter() - start_time

        self.assertIsNone(result)
        self.assertLess(elapsed, 0.15, f"El timeout tardó demasiado: {elapsed:.3f}s")

    # =========================================================================
    # 5. Manejo de Excepciones del Cliente / Red
    # =========================================================================

    def test_analyze_setup_api_client_error(self):
        """Verifica que errores de API (429, 403, 500) retornan None sin propagar excepciones."""
        analyst = GeminiAnalyst(api_key="mock_key", timeout=5.0)

        mock_client = MagicMock()
        mock_aio = MagicMock()
        mock_models = MagicMock()
        mock_models.generate_content = AsyncMock(side_effect=Exception("429 ResourceExhausted: Quota exceeded"))
        mock_aio.models = mock_models
        mock_client.aio = mock_aio
        analyst.client = mock_client

        result = asyncio.run(analyst.analyze_setup(self.valid_setup))
        self.assertIsNone(result)

    def test_analyze_setup_network_disconnect_error(self):
        """Verifica que caídas de conexión o socket cerrado retornan None."""
        analyst = GeminiAnalyst(api_key="mock_key", timeout=5.0)

        mock_client = MagicMock()
        mock_aio = MagicMock()
        mock_models = MagicMock()
        mock_models.generate_content = AsyncMock(side_effect=ConnectionResetError("Remote host closed connection"))
        mock_aio.models = mock_models
        mock_client.aio = mock_aio
        analyst.client = mock_client

        result = asyncio.run(analyst.analyze_setup(self.valid_setup))
        self.assertIsNone(result)

    def test_analyze_setup_safety_blocked_response(self):
        """Verifica respuesta bloqueada por filtros de seguridad donde .text es None o lanza ValueError."""
        analyst = GeminiAnalyst(api_key="mock_key", timeout=5.0)

        # Caso A: response.text es None
        mock_resp_a = MagicMock()
        mock_resp_a.text = None

        # Caso B: response.text genera ValueError (típico de google-genai cuando finish_reason=SAFETY)
        mock_resp_b = MagicMock()
        type(mock_resp_b).text = property(lambda self: (_ for _ in ()).throw(ValueError("Safety filter triggered")))

        for mock_resp in [mock_resp_a, mock_resp_b]:
            mock_client = MagicMock()
            mock_aio = MagicMock()
            mock_models = MagicMock()
            mock_models.generate_content = AsyncMock(return_value=mock_resp)
            mock_aio.models = mock_models
            mock_client.aio = mock_aio
            analyst.client = mock_client

            result = asyncio.run(analyst.analyze_setup(self.valid_setup))
            self.assertIsNone(result)

    def test_analyze_setup_successful_cleaning(self):
        """Verifica que una respuesta exitosa con comillas sea limpiada correctamente."""
        analyst = GeminiAnalyst(api_key="mock_key", timeout=5.0)

        mock_resp = MagicMock()
        mock_resp.text = '  "Estructura alcista sólida en BTCUSDT tras barrido de liquidez. El Order Block en $64,000 ofrece soporte institucional clave. Con R:R de 1:3.0 y 85% de probabilidad ML, la operación presenta alta asimetría. Se recomienda mantener disciplina de riesgo con stop loss estricto."  '

        mock_client = MagicMock()
        mock_aio = MagicMock()
        mock_models = MagicMock()
        mock_models.generate_content = AsyncMock(return_value=mock_resp)
        mock_aio.models = mock_models
        mock_client.aio = mock_aio
        analyst.client = mock_client

        result = asyncio.run(analyst.analyze_setup(self.valid_setup))
        self.assertIsNotNone(result)
        self.assertFalse(result.startswith('"'))
        self.assertFalse(result.endswith('"'))
        self.assertIn("Estructura alcista sólida", result)

    # =========================================================================
    # 6. Concurrencia y Carga Paralela
    # =========================================================================

    def test_concurrent_analyze_setups(self):
        """Verifica ejecución concurrente de 10 llamadas simultáneas sin colisiones ni condiciones de carrera."""
        analyst = GeminiAnalyst(api_key="mock_key", timeout=5.0)

        async def mock_generate(model, contents, config):
            await asyncio.sleep(0.01)
            symbol = "UNKNOWN"
            for line in contents.splitlines():
                if "Símbolo:" in line:
                    symbol = line.split("|")[0].split(":")[1].strip()
                    break
            mock_resp = MagicMock()
            mock_resp.text = f"Análisis institucional confirmado para {symbol}."
            return mock_resp

        mock_client = MagicMock()
        mock_aio = MagicMock()
        mock_models = MagicMock()
        mock_models.generate_content = mock_generate
        mock_aio.models = mock_models
        mock_client.aio = mock_aio
        analyst.client = mock_client

        async def run_batch():
            symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "SUIUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "BNBUSDT"]
            tasks = []
            for s in symbols:
                setup = TradeSetup(
                    symbol=s,
                    direction="LONG",
                    entry_price=100.0,
                    stop_loss=95.0,
                    tp1=105.0,
                    tp2=110.0,
                    tp3=120.0,
                    risk_distance=5.0,
                    rr_tp1=1.0,
                    rr_tp2=2.0,
                    rr_tp3=4.0,
                    confidence_score=80.0,
                    reasons=["Test Reason"]
                )
                tasks.append(analyst.analyze_setup(setup))
            return await asyncio.gather(*tasks)

        results = asyncio.run(run_batch())
        self.assertEqual(len(results), 10)
        for i, res in enumerate(results):
            self.assertIsNotNone(res)
            self.assertIn("Análisis institucional", res)

    # =========================================================================
    # 7. Resiliencia Extrema: Setup Malformado y setup.symbol
    # =========================================================================

    def test_malformed_setup_resilience(self):
        """Verifica qué ocurre si setup tiene atributos None o extremos mientras está configurado."""
        analyst = GeminiAnalyst(api_key="mock_key", timeout=5.0)

        # Mockear client
        mock_client = MagicMock()
        mock_aio = MagicMock()
        mock_models = MagicMock()
        mock_models.generate_content = AsyncMock(return_value=MagicMock(text="Análisis OK"))
        mock_aio.models = mock_models
        mock_client.aio = mock_aio
        analyst.client = mock_client

        # Setup con valores numéricos None o extraños
        corrupt_setup = MagicMock()
        corrupt_setup.symbol = "CORRUPT"
        corrupt_setup.direction = "LONG"
        corrupt_setup.entry_price = None
        corrupt_setup.stop_loss = None
        corrupt_setup.risk_distance = None
        corrupt_setup.tp1 = None
        corrupt_setup.rr_tp1 = 1.0
        corrupt_setup.tp2 = None
        corrupt_setup.rr_tp2 = 2.0
        corrupt_setup.tp3 = None
        corrupt_setup.rr_tp3 = 3.0
        corrupt_setup.confidence_score = 75.0
        corrupt_setup.reasons = None

        res = asyncio.run(analyst.analyze_setup(corrupt_setup))
        self.assertIsNotNone(res)
        self.assertEqual(res, "Análisis OK")

    def test_setup_is_none_or_missing_symbol_attribute(self):
        """
        PRUEBA ADVERSARIAL CRÍTICA:
        Si setup es None o no tiene el atributo symbol, ¿escapa alguna excepción
        dentro del bloque except de analyze_setup al intentar registrar setup.symbol?
        """
        analyst = GeminiAnalyst(api_key="mock_key", timeout=5.0)
        mock_client = MagicMock()
        analyst.client = mock_client

        # Si pasamos setup=None a analyze_setup:
        # En analyze_setup: features_summary = self._format_features(features) -> OK
        # prompt = self._build_prompt(setup, ...) -> setup.reasons -> AttributeError: 'NoneType' object has no attribute 'reasons'
        # Luego en except: logger.warning("Error ... %s", setup.symbol, e) -> setup.symbol -> AttributeError!
        try:
            res = asyncio.run(analyst.analyze_setup(None))
            # Si no lanza excepción, perfecto
            self.assertIsNone(res)
        except Exception as exc:
            # Si lanza excepción, registramos la falla empíricamente
            self.fail(f"Falla de blindaje: analyze_setup(None) lanzó una excepción no controlada: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    unittest.main()

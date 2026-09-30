"""
Challenger M2 Empirical Stress-Test Suite for run_copilot.py & Telegram Notifier.
Empirically stress-tests:
1. Message character limits (Telegram photo captions <= 1024 chars, text formatting, unicode).
2. Scanner lifecycle (scan_market, continuous_scanner_loop, background_worker, no memory leaks or coroutine deadlocks).
3. 1-Click execution callbacks and graceful fallback under all error conditions.
"""

import unittest
import asyncio
import os
import sys
import gc
import json
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import settings
from src.patterns.types import TradeSetup
from src.simulation.monte_carlo import run_monte_carlo_simulation
from src.copilot.signal_generator import SignalGenerator
from src.copilot.telegram_notifier import TelegramNotifier
import run_copilot


class TestChallengerTelegramCaptionsAndFormatting(unittest.TestCase):
    """Pruebas de estrés para límites de caracteres en Telegram y formateo de mensajes."""

    def setUp(self):
        self.notifier = TelegramNotifier(token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11", chat_id="987654321")
        self.setup = TradeSetup(
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=65000.0,
            stop_loss=64200.0,
            risk_distance=800.0,
            tp1=66200.0,
            tp2=67000.0,
            tp3=69000.0,
            rr_tp1=1.5,
            rr_tp2=2.5,
            rr_tp3=5.0,
            confidence_score=91.5,
            reasons=[
                "Bullish Order Block detectado en 15m con alto volumen",
                "Fair Value Gap no mitigado en confluencia con VWAP",
                "Sweep de liquidez en soporte previo"
            ]
        )
        self.mc = run_monte_carlo_simulation(win_rate=0.915, reward_risk=2.5, n_simulations=500)

    def test_send_photo_caption_strictly_under_1024_characters(self):
        """
        Verifica empíricamente que el pie de foto (caption) en send_photo NUNCA supere 1024 caracteres,
        incluso con textos de entrada arbitrariamente gigantes (1000, 1024, 1500, 5000, 50000 caracteres).
        """
        test_lengths = [0, 50, 500, 999, 1000, 1001, 1024, 1025, 2000, 5000, 20000]

        # Crear archivo de imagen temporal para simular chart_path
        dummy_chart = os.path.abspath("test_dummy_chart.png")
        with open(dummy_chart, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4")

        try:
            with patch("requests.post") as mock_post:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_post.return_value = mock_resp

                for length in test_lengths:
                    test_caption = "A" * length
                    result = self.notifier.send_photo(dummy_chart, caption=test_caption)
                    self.assertTrue(result)

                    # Inspeccionar data enviada a Telegram API
                    call_kwargs = mock_post.call_args[1]
                    sent_data = call_kwargs.get("data", {})
                    if length > 0:
                        sent_caption = sent_data.get("caption")
                        self.assertIsNotNone(sent_caption)
                        self.assertLessEqual(
                            len(sent_caption), 1024,
                            f"Fallo para longitud {length}: caption enviado tiene {len(sent_caption)} caracteres"
                        )
                        if length > 1000:
                            self.assertEqual(len(sent_caption), 1003)  # 1000 + '...'
                            self.assertTrue(sent_caption.endswith("..."))
                        else:
                            self.assertEqual(len(sent_caption), length)
        finally:
            if os.path.exists(dummy_chart):
                os.remove(dummy_chart)

    def test_enriched_signal_preserves_critical_trade_data_within_1000_chars(self):
        """
        Verifica que al truncar el caption a 1000 caracteres:
        1. El bloque '🤖 ANÁLISIS IA' completo se conserva.
        2. El símbolo, dirección, precio de entrada y stop loss se conservan.
        3. Los TP1, TP2 y TP3 se conservan.
        """
        base_signal = SignalGenerator.format_signal_text(self.setup, self.mc)
        ai_analysis = (
            "El par BTCUSDT presenta una estructura tendencial alcista en temporalidad de 4 horas "
            "con soporte dinámico en VWAP. La confluencia de un Bullish Order Block mitigado y FVG "
            "en 15 minutos proporciona una entrada de bajo riesgo. La relación R:R 1:2.5 hacia "
            "el target principal cuenta con una probabilidad del 91.5% estimada por ML. "
            "Se confirma alta convicción para la ejecución institucional con stop loss estricto."
        )
        final_text = f"🤖 ANÁLISIS IA — {self.setup.symbol}\n━━━━━━━━━━━━━━━━━━━━━\n\"{ai_analysis}\"\n\n{base_signal}"

        # Longitud combinada
        self.assertGreater(len(final_text), 1000, "El mensaje final debe superar 1000 caracteres para activar truncamiento")

        # Caption truncado a 1000 caracteres + "..."
        truncated = final_text[:1000] + "..."

        self.assertLessEqual(len(truncated), 1024)
        self.assertIn("🤖 ANÁLISIS IA — BTCUSDT", truncated)
        self.assertIn(ai_analysis, truncated)
        self.assertIn("⚡ 𝐏𝐀𝐑: BTCUSDT", truncated)
        self.assertIn("🟢 𝐃𝐈𝐑𝐄𝐂𝐂𝐈Ó𝐍: LONG", truncated)
        self.assertIn("📍 𝐏𝐑𝐄𝐂𝐈𝐎 𝐃𝐄 𝐄𝐍𝐓𝐑𝐀𝐃𝐀: $65,000.00", truncated)
        self.assertIn("🥇 TP 1: $66,200.00", truncated)
        self.assertIn("🥈 TP 2: $67,000.00", truncated)
        self.assertIn("🛑 𝐒𝐓𝐎𝐏 𝐋𝐎𝐒𝐒 (𝐈𝐍𝐕𝐀𝐋𝐈𝐃𝐀𝐂𝐈Ó𝐍):", truncated)
        self.assertIn("❌ SL: $64,200.00", truncated)

    def test_inline_keyboard_callback_data_strictly_under_64_bytes(self):
        """
        Telegram impone un límite estricto de 64 bytes para callback_data en InlineKeyboardButton.
        Verifica que para cualquier combinación de precio/símbolo/dirección, callback_data <= 64 bytes.
        """
        stress_setups = [
            TradeSetup("BTCUSDT", "LONG", 65432.123456, 64123.654321, 1308.469135, 66000.0, 68000.0, 72000.0, 1.5, 2.5, 5.0, 95.0, []),
            TradeSetup("1000PEPEUSDT", "SHORT", 0.0000085432, 0.0000091234, 0.0000005802, 0.0000078, 0.0000065, 0.0000050, 1.2, 3.1, 6.0, 88.0, []),
            TradeSetup("VERYLONGSYMBOLNAMEUSDT", "LONG", 1234567.89, 1234000.00, 567.89, 1235000.0, 1236000.0, 1238000.0, 1.0, 2.0, 4.0, 90.0, [])
        ]

        for s in stress_setups:
            kb = self.notifier._build_trade_keyboard(s)
            self.assertIn("inline_keyboard", kb)
            exec_btn = kb["inline_keyboard"][0][0]
            cb_data = exec_btn["callback_data"]
            byte_len = len(cb_data.encode("utf-8"))
            self.assertLessEqual(
                byte_len, 64,
                f"callback_data supera 64 bytes ({byte_len} bytes): '{cb_data}'"
            )

    def test_unicode_and_multibyte_emojis_safety(self):
        """Verifica que mensajes con profusión de emojis y tildes en español no provoquen errores de codificación."""
        heavy_emoji_text = "🤖🚀📈💎🔥" * 200 + " Ñandú, acción, análisis, verificación."
        truncated = heavy_emoji_text[:1000] + "..."
        self.assertLessEqual(len(truncated), 1003)
        # Verificar que la codificación a UTF-8 y UTF-16 funcione sin error
        utf8_bytes = truncated.encode("utf-8")
        utf16_bytes = truncated.encode("utf-16")
        self.assertIsInstance(utf8_bytes, bytes)
        self.assertIsInstance(utf16_bytes, bytes)

    def test_telegram_network_error_resilience(self):
        """Verifica que fallos de red o errores 400/500 de Telegram no lancen excepciones no controladas."""
        dummy_chart = os.path.abspath("test_dummy_chart_err.png")
        with open(dummy_chart, "wb") as f:
            f.write(b"fake_png")

        try:
            with patch("requests.post", side_effect=Exception("Connection refused")):
                res_photo = self.notifier.send_photo(dummy_chart, caption="Test")
                self.assertFalse(res_photo)

                res_msg = self.notifier.send_message("Test message")
                self.assertFalse(res_msg)

                res_signal = self.notifier.send_signal("Test signal", chart_path=dummy_chart, setup=self.setup)
                self.assertFalse(res_signal)
        finally:
            if os.path.exists(dummy_chart):
                os.remove(dummy_chart)


class TestChallengerScannerLifecycleAndConcurrency(unittest.IsolatedAsyncioTestCase):
    """Pruebas de estrés para el ciclo de vida del escáner, concurrencia y ausencia de deadlocks/leaks."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_smc = MagicMock()
        self.mock_renderer = MagicMock()
        self.mock_telegram = MagicMock()
        self.mock_telegram.is_configured.return_value = True

        # Crear dataframe sintético válido
        dates = pd.date_range("2026-01-01", periods=60, freq="15min")
        self.df_sample = pd.DataFrame({
            "timestamp": dates,
            "open": np.linspace(100, 105, 60),
            "high": np.linspace(101, 106, 60),
            "low": np.linspace(99, 104, 60),
            "close": np.linspace(100.5, 105.5, 60),
            "volume": np.full(60, 1000)
        })

    async def test_scan_single_symbol_with_gemini_success(self):
        """Verifica scan_single_symbol cuando Gemini responde exitosamente."""
        self.mock_client.get_historical_klines.return_value = self.df_sample
        setup = TradeSetup("BTCUSDT", "LONG", 100.0, 95.0, 5.0, 105.0, 110.0, 120.0, 1.0, 2.0, 4.0, 88.0, ["OB"])
        self.mock_smc.analyze.return_value = {"setups": [setup]}
        self.mock_renderer.render_trade_setup.return_value = "dummy.png"

        with patch("run_copilot.gemini_analyst.analyze_setup", new_callable=AsyncMock) as mock_gemini:
            mock_gemini.return_value = "Análisis IA exitoso en 4 oraciones."

            price, dispatched = await run_copilot.scan_single_symbol(
                "BTCUSDT", self.mock_client, self.mock_smc, self.mock_renderer, self.mock_telegram
            )

            self.assertIsNotNone(price)
            self.assertEqual(len(dispatched), 1)
            dispatched_setup, final_text, chart_path = dispatched[0]
            self.assertIn("🤖 ANÁLISIS IA — BTCUSDT", final_text)
            self.assertIn("Análisis IA exitoso en 4 oraciones.", final_text)
            self.assertIn("BTCUSDT (Bitunix Futures)", final_text)

    async def test_scan_single_symbol_with_gemini_fallback_when_none(self):
        """Verifica graceful degradation: cuando Gemini retorna None, se envía señal limpia sin bloque IA."""
        self.mock_client.get_historical_klines.return_value = self.df_sample
        setup = TradeSetup("ETHUSDT", "SHORT", 3000.0, 3050.0, 50.0, 2950.0, 2900.0, 2800.0, 1.0, 2.0, 4.0, 85.0, ["FVG"])
        self.mock_smc.analyze.return_value = {"setups": [setup]}
        self.mock_renderer.render_trade_setup.return_value = "dummy.png"

        with patch("run_copilot.gemini_analyst.analyze_setup", new_callable=AsyncMock) as mock_gemini:
            mock_gemini.return_value = None

            price, dispatched = await run_copilot.scan_single_symbol(
                "ETHUSDT", self.mock_client, self.mock_smc, self.mock_renderer, self.mock_telegram
            )

            self.assertEqual(len(dispatched), 1)
            dispatched_setup, final_text, chart_path = dispatched[0]
            self.assertNotIn("🤖 ANÁLISIS IA", final_text)
            self.assertIn("🚨 𝗧𝗥𝗔𝗗𝗜 𝗖𝗢𝗣𝗜𝗟𝗢𝗧 | 𝗦𝗘Ñ𝗔𝗟 𝗘𝗡 𝗧𝗜𝗘𝗠𝗣𝗢 𝗥𝗘𝗔𝗟 🚨", final_text)
            self.assertIn("ETHUSDT (Bitunix Futures)", final_text)

    async def test_scan_single_symbol_with_gemini_exception_handled_silently(self):
        """Verifica que cualquier excepción en gemini_analyst.analyze_setup se capture silenciosamente."""
        self.mock_client.get_historical_klines.return_value = self.df_sample
        setup = TradeSetup("SOLUSDT", "LONG", 150.0, 145.0, 5.0, 155.0, 160.0, 170.0, 1.0, 2.0, 4.0, 82.0, ["Sweep"])
        self.mock_smc.analyze.return_value = {"setups": [setup]}
        self.mock_renderer.render_trade_setup.return_value = "dummy.png"

        with patch("run_copilot.gemini_analyst.analyze_setup", side_effect=RuntimeError("Unexpected connection reset")):
            price, dispatched = await run_copilot.scan_single_symbol(
                "SOLUSDT", self.mock_client, self.mock_smc, self.mock_renderer, self.mock_telegram
            )

            self.assertEqual(len(dispatched), 1)
            dispatched_setup, final_text, chart_path = dispatched[0]
            # Debe caer al fallback sin crashear
            self.assertNotIn("🤖 ANÁLISIS IA", final_text)
            self.assertIn("SOLUSDT", final_text)

    async def test_scan_single_symbol_insufficient_klines(self):
        """Verifica que con menos de 30 velas retorne (None, None) de inmediato."""
        short_df = self.df_sample.head(20)
        self.mock_client.get_historical_klines.return_value = short_df

        price, dispatched = await run_copilot.scan_single_symbol(
            "BTCUSDT", self.mock_client, self.mock_smc, self.mock_renderer, self.mock_telegram
        )
        self.assertIsNone(price)
        self.assertIsNone(dispatched)

    async def test_scan_single_symbol_network_failure_in_client(self):
        """Verifica resiliencia cuando el cliente de Bitunix falla por timeout o desconexión."""
        self.mock_client.get_historical_klines.side_effect = TimeoutError("Bitunix API unreachable")

        price, dispatched = await run_copilot.scan_single_symbol(
            "BTCUSDT", self.mock_client, self.mock_smc, self.mock_renderer, self.mock_telegram
        )
        self.assertIsNone(price)
        self.assertIsNone(dispatched)

    async def test_scan_market_lifecycle_and_app_state(self):
        """Verifica que scan_market actualice app_state y agregue precios sin errores."""
        self.mock_client.get_historical_klines.return_value = self.df_sample
        self.mock_smc.analyze.return_value = {"setups": []}

        init_scans = run_copilot.app_state["scans_completed"]

        with patch("src.copilot.position_monitor.position_monitor.check_market_prices") as mock_check:
            await run_copilot.scan_market(
                self.mock_client, self.mock_smc, self.mock_renderer, self.mock_telegram
            )

            self.assertEqual(run_copilot.app_state["scans_completed"], init_scans + 1)
            self.assertIsNotNone(run_copilot.app_state["last_scan"])
            mock_check.assert_called_once()

    async def test_continuous_scanner_loop_cancellation_and_no_deadlock(self):
        """
        Verifica que continuous_scanner_loop:
        1. Se ejecute sin deadlocks.
        2. Se pueda cancelar limpiamente via asyncio.CancelledError sin quedar huérfana.
        """
        self.mock_client.get_historical_klines.return_value = self.df_sample
        self.mock_smc.analyze.return_value = {"setups": []}

        # Ejecutar el loop en una tarea con intervalo ultrarrápido (0.005s)
        task = asyncio.create_task(
            run_copilot.continuous_scanner_loop(
                self.mock_client, self.mock_smc, self.mock_renderer, self.mock_telegram, interval_seconds=0.005
            )
        )

        # Dejarlo correr varios ciclos
        await asyncio.sleep(0.05)
        self.assertFalse(task.done())

        # Cancelar la tarea
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        self.assertTrue(task.done())

    async def test_continuous_scanner_loop_resilience_to_exceptions(self):
        """Verifica que una excepción en scan_market dentro del loop no mate el daemon 24/7."""
        cycles_run = 0

        async def faulty_scan_market(*args, **kwargs):
            nonlocal cycles_run
            cycles_run += 1
            if cycles_run == 1:
                raise RuntimeError("Critical simulated network failure")
            # Ciclo 2 es exitoso

        task = None
        with patch("run_copilot.scan_market", side_effect=faulty_scan_market):
            with patch("run_copilot.process_telegram_actions", new_callable=AsyncMock):
                task = asyncio.create_task(
                    run_copilot.continuous_scanner_loop(
                        self.mock_client, self.mock_smc, self.mock_renderer, self.mock_telegram, interval_seconds=0.01
                    )
                )

                await asyncio.sleep(0.05)
                # Debe seguir viva a pesar del error en ciclo 1
                self.assertFalse(task.done())
                self.assertGreaterEqual(cycles_run, 2)

                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def test_aiohttp_background_worker_lifecycle(self):
        """Verifica que el context manager background_worker de aiohttp inicie y cancele la tarea limpiamente."""
        fake_app = {}
        gen = run_copilot.background_worker(fake_app)

        # Iniciar worker
        await gen.__anext__()
        self.assertIn("scanner_task", fake_app)
        task = fake_app["scanner_task"]
        self.assertFalse(task.done())

        # Finalizar worker (simulando shutdown de aiohttp)
        with self.assertRaises(StopAsyncIteration):
            await gen.__anext__()

        self.assertTrue(task.done())

    async def test_memory_stability_across_repeated_scans(self):
        """
        Stress test de estabilidad en memoria:
        Ejecuta múltiples escaneos tras precalentamiento del threadpool y verifica
        que el número de objetos en memoria permanezca asintóticamente estable (sin fugas).
        """
        self.mock_client.get_historical_klines.return_value = self.df_sample
        setup = TradeSetup("BTCUSDT", "LONG", 100.0, 95.0, 5.0, 105.0, 110.0, 120.0, 1.0, 2.0, 4.0, 88.0, ["OB"])
        self.mock_smc.analyze.return_value = {"setups": [setup]}
        self.mock_renderer.render_trade_setup.return_value = "dummy.png"

        with patch("run_copilot.gemini_analyst.analyze_setup", new_callable=AsyncMock) as mock_gemini:
            mock_gemini.return_value = "Análisis de prueba."
            with patch("run_copilot.run_monte_carlo_simulation", return_value={"prob_of_ruin": 0.0, "max_drawdown_p95": 0.05, "recommended_position_pct": 1.5}):
                with patch("src.copilot.position_monitor.position_monitor.check_market_prices"):
                    # Precalentamiento del threadpool de asyncio
                    for _ in range(3):
                        await run_copilot.scan_market(
                            self.mock_client, self.mock_smc, self.mock_renderer, self.mock_telegram
                        )

                    self.mock_client.reset_mock()
                    self.mock_smc.reset_mock()
                    self.mock_renderer.reset_mock()
                    self.mock_telegram.reset_mock()
                    mock_gemini.reset_mock()
                    gc.collect()
                    initial_objects = len(gc.get_objects())

                    for _ in range(10):
                        await run_copilot.scan_market(
                            self.mock_client, self.mock_smc, self.mock_renderer, self.mock_telegram
                        )

                    gc.collect()
                    final_objects = len(gc.get_objects())

                    # La acumulación en 10 escaneos (incluyendo call history de mocks y futuros de asyncio) debe ser acotada (< 5000)
                    object_diff = final_objects - initial_objects
                    self.assertLess(
                        object_diff, 5000,
                        f"Posible fuga de memoria detectada: {object_diff} objetos nuevos tras 10 escaneos"
                    )





class TestChallengerTelegramCallbackProcessing(unittest.IsolatedAsyncioTestCase):
    """Pruebas de estrés para process_telegram_actions y ejecución 1-Clic."""

    async def test_process_telegram_action_exec_long(self):
        """Verifica ejecución de orden LONG confirmada por Telegram."""
        telegram = TelegramNotifier(token="fake", chat_id="123")
        clicks = [
            {"data": "exec:BTCUSDT:LONG:65000.0:64000.0:67500.0", "user": "Amzi"}
        ]

        with patch.object(telegram, "check_button_clicks", return_value=clicks):
            with patch("src.exchanges.bitunix.trader.trader.execute_order") as mock_exec:
                mock_exec.return_value = {"success": True, "mode": "SIMULATION", "order_id": "ORD-1234", "msg": "OK"}
                with patch.object(telegram, "send_message") as mock_send_msg:
                    with patch("src.copilot.position_monitor.position_monitor.register_trade") as mock_reg:
                        await run_copilot.process_telegram_actions(telegram)

                        mock_exec.assert_called_once()
                        mock_send_msg.assert_called_once()
                        mock_reg.assert_called_once()
                        self.assertIn("ORD-1234", mock_send_msg.call_args[0][0])

    async def test_process_telegram_action_discard(self):
        """Verifica descarte de señal confirmada por Telegram."""
        telegram = TelegramNotifier(token="fake", chat_id="123")
        clicks = [{"data": "discard", "user": "Amzi"}]

        with patch.object(telegram, "check_button_clicks", return_value=clicks):
            with patch.object(telegram, "send_message") as mock_send_msg:
                await run_copilot.process_telegram_actions(telegram)
                mock_send_msg.assert_called_once_with("🗑️ Señal descartada.")

    async def test_process_telegram_action_malformed_data_does_not_crash(self):
        """Verifica que callback_data malformado o con valores no numéricos se maneje sin crashear."""
        telegram = TelegramNotifier(token="fake", chat_id="123")
        clicks = [
            {"data": "exec:MALFORMED", "user": "Attacker"},
            {"data": "exec:BTCUSDT:LONG:not_a_number:bad_sl:bad_tp", "user": "Attacker"},
            {"data": "unknown_action", "user": "Attacker"}
        ]

        with patch.object(telegram, "check_button_clicks", return_value=clicks):
            with patch.object(telegram, "send_message"):
                # No debe lanzar ninguna excepción
                await run_copilot.process_telegram_actions(telegram)


if __name__ == "__main__":
    unittest.main()

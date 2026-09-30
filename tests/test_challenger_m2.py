"""
Empirical Challenger Test Suite for Milestone M2:
Adversarial Verification of Copilot Telegram Integration in run_copilot.py.

Tests scan_single_symbol across:
- Case A: Gemini returns valid analysis string (enriched message format)
- Case B: Gemini returns None or empty/whitespace (graceful degradation to original signal)
- Case C: Gemini raises exceptions (network error, timeout, API failure) -> safe fallback
- Case D: 1-Click buttons keyboard object continuity across send_signal and notify_signal
- Case E: Multi-setup dispatch, unconfigured Telegram, and market data edge cases
"""

import sys
import os
import unittest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.patterns.types import TradeSetup
from src.copilot.telegram_notifier import TelegramNotifier
import run_copilot


def create_dummy_df(n: int = 50) -> pd.DataFrame:
    """Generates synthetic OHLCV dataframe with sufficient candles for scanning."""
    dates = pd.date_range("2026-01-01", periods=n, freq="15min")
    close = np.linspace(60000, 65000, n)
    return pd.DataFrame({
        "timestamp": dates,
        "open": close - 50.0,
        "high": close + 100.0,
        "low": close - 100.0,
        "close": close,
        "volume": np.full(n, 150.0)
    })


def create_dummy_setup(symbol: str = "BTCUSDT") -> TradeSetup:
    """Creates a realistic TradeSetup instance for testing."""
    return TradeSetup(
        symbol=symbol,
        direction="LONG",
        entry_price=64000.0,
        stop_loss=63000.0,
        tp1=65500.0,
        tp2=66500.0,
        tp3=68000.0,
        risk_distance=1000.0,
        rr_tp1=1.5,
        rr_tp2=2.5,
        rr_tp3=4.0,
        confidence_score=88.0,
        reasons=["Mitigación de Bullish Order Block", "Sweep de liquidez previa"]
    )


class TestChallengerM2CopilotTelegram(unittest.TestCase):
    """Adversarial stress and verification suite for run_copilot.py scan_single_symbol."""

    def setUp(self):
        self.df_mtf = create_dummy_df(60)
        self.df_htf = create_dummy_df(60)
        self.setup = create_dummy_setup("BTCUSDT")
        self.expected_signal_text = "🟢 SEÑAL DE COMPRA: BTCUSDT\nEntrada: $64,000.00\nSL: $63,000.00\nTP2: $66,500.00"

        # Mock Bitunix Client
        self.mock_client = MagicMock()
        self.mock_client.get_historical_klines.side_effect = lambda sym, interval, limit: self.df_mtf if "15m" in interval or limit > 60 else self.df_htf

        # Mock SMC Engine
        self.mock_smc = MagicMock()
        self.mock_smc.analyze.return_value = {"setups": [self.setup]}

        # Mock Chart Renderer
        self.mock_renderer = MagicMock()
        self.mock_renderer.render_trade_setup.return_value = "output/charts/test_btcusdt.png"

    # =========================================================================
    # Group A: Case A — Gemini returns a valid analysis string
    # =========================================================================

    def test_case_a_valid_analysis_enriches_telegram_message(self):
        """
        Case A: Gemini returns valid analysis.
        Assert message contains '🤖 ANÁLISIS IA — {SYMBOL}', quotes, and original signal.
        """
        ai_text = "Confluencia sólida en Order Block H1 con Sweep de SSL. Veredicto: Compra con alta probabilidad."
        mock_telegram = MagicMock(spec=TelegramNotifier)
        mock_telegram.is_configured.return_value = True

        async def run_test():
            with patch("run_copilot.SignalGenerator.format_signal_text", return_value=self.expected_signal_text), \
                 patch("run_copilot.gemini_analyst.analyze_setup", new=AsyncMock(return_value=ai_text)):
                
                price, dispatched = await run_copilot.scan_single_symbol(
                    "BTCUSDT", self.mock_client, self.mock_smc, self.mock_renderer, mock_telegram
                )
                return price, dispatched

        price, dispatched = asyncio.run(run_test())

        self.assertIsNotNone(price)
        self.assertEqual(len(dispatched), 1)
        s, final_text, chart_path = dispatched[0]

        # Verify format components
        expected_header = "🤖 ANÁLISIS IA — BTCUSDT\n━━━━━━━━━━━━━━━━━━━━━\n"
        self.assertIn(expected_header, final_text)
        self.assertIn(f'"{ai_text}"', final_text)
        self.assertTrue(final_text.endswith(self.expected_signal_text))
        self.assertEqual(chart_path, "output/charts/test_btcusdt.png")

        # Verify telegram was called with final_text
        mock_telegram.send_signal.assert_called_once_with(
            final_text, "output/charts/test_btcusdt.png", setup=self.setup
        )

    def test_case_a_whitespace_stripped_properly(self):
        """Case A variation: Analysis text with leading/trailing whitespaces and tabs is cleanly stripped."""
        raw_ai_text = "\n\t  Análisis con espacios en bordes. Confianza 90%.  \r\n\t "
        mock_telegram = MagicMock(spec=TelegramNotifier)
        mock_telegram.is_configured.return_value = True

        async def run_test():
            with patch("run_copilot.SignalGenerator.format_signal_text", return_value=self.expected_signal_text), \
                 patch("run_copilot.gemini_analyst.analyze_setup", new=AsyncMock(return_value=raw_ai_text)):
                
                return await run_copilot.scan_single_symbol(
                    "BTCUSDT", self.mock_client, self.mock_smc, self.mock_renderer, mock_telegram
                )

        _, dispatched = asyncio.run(run_test())
        final_text = dispatched[0][1]

        # Quoted content should not contain leading/trailing whitespace
        self.assertIn('"Análisis con espacios en bordes. Confianza 90%."', final_text)
        self.assertNotIn('"\n\t  ', final_text)

    def test_case_a_unicode_and_emojis_preserved(self):
        """Case A variation: Analysis text with diverse Unicode characters and emojis formats cleanly."""
        ai_text = "⚡ Entrada institucional confirmada en $64,000 🚀. FVG respetado al 100% 🎯."
        mock_telegram = MagicMock(spec=TelegramNotifier)
        mock_telegram.is_configured.return_value = True

        async def run_test():
            with patch("run_copilot.SignalGenerator.format_signal_text", return_value=self.expected_signal_text), \
                 patch("run_copilot.gemini_analyst.analyze_setup", new=AsyncMock(return_value=ai_text)):
                
                return await run_copilot.scan_single_symbol(
                    "BTCUSDT", self.mock_client, self.mock_smc, self.mock_renderer, mock_telegram
                )

        _, dispatched = asyncio.run(run_test())
        final_text = dispatched[0][1]
        self.assertIn(f'"{ai_text}"', final_text)

    # =========================================================================
    # Group B: Case B — Gemini returns None, empty, or whitespace
    # =========================================================================

    def test_case_b_none_falls_back_to_original_signal(self):
        """
        Case B: Gemini returns None.
        Assert Telegram message equals the original signal text with no modification.
        """
        mock_telegram = MagicMock(spec=TelegramNotifier)
        mock_telegram.is_configured.return_value = True

        async def run_test():
            with patch("run_copilot.SignalGenerator.format_signal_text", return_value=self.expected_signal_text), \
                 patch("run_copilot.gemini_analyst.analyze_setup", new=AsyncMock(return_value=None)):
                
                return await run_copilot.scan_single_symbol(
                    "BTCUSDT", self.mock_client, self.mock_smc, self.mock_renderer, mock_telegram
                )

        price, dispatched = asyncio.run(run_test())
        self.assertEqual(len(dispatched), 1)
        final_text = dispatched[0][1]

        # Message must be exactly the original signal_text
        self.assertEqual(final_text, self.expected_signal_text)
        self.assertNotIn("🤖 ANÁLISIS IA", final_text)

        mock_telegram.send_signal.assert_called_once_with(
            self.expected_signal_text, "output/charts/test_btcusdt.png", setup=self.setup
        )

    def test_case_b_empty_string_falls_back_to_original_signal(self):
        """Case B variation: Gemini returns empty string -> graceful fallback to original signal."""
        mock_telegram = MagicMock(spec=TelegramNotifier)
        mock_telegram.is_configured.return_value = True

        async def run_test():
            with patch("run_copilot.SignalGenerator.format_signal_text", return_value=self.expected_signal_text), \
                 patch("run_copilot.gemini_analyst.analyze_setup", new=AsyncMock(return_value="")):
                
                return await run_copilot.scan_single_symbol(
                    "BTCUSDT", self.mock_client, self.mock_smc, self.mock_renderer, mock_telegram
                )

        _, dispatched = asyncio.run(run_test())
        self.assertEqual(dispatched[0][1], self.expected_signal_text)

    def test_case_b_whitespace_only_falls_back_to_original_signal(self):
        """Case B variation: Gemini returns whitespace-only string -> graceful fallback."""
        mock_telegram = MagicMock(spec=TelegramNotifier)
        mock_telegram.is_configured.return_value = True

        async def run_test():
            with patch("run_copilot.SignalGenerator.format_signal_text", return_value=self.expected_signal_text), \
                 patch("run_copilot.gemini_analyst.analyze_setup", new=AsyncMock(return_value="   \t\n   ")):
                
                return await run_copilot.scan_single_symbol(
                    "BTCUSDT", self.mock_client, self.mock_smc, self.mock_renderer, mock_telegram
                )

        _, dispatched = asyncio.run(run_test())
        self.assertEqual(dispatched[0][1], self.expected_signal_text)

    # =========================================================================
    # Group C: Case C — Gemini raises exception
    # =========================================================================

    def test_case_c_simulated_network_failure_fallback(self):
        """
        Case C: Gemini raises ConnectionError.
        Assert fallback to original signal text with no uncaught exception.
        """
        mock_telegram = MagicMock(spec=TelegramNotifier)
        mock_telegram.is_configured.return_value = True

        async def run_test():
            with patch("run_copilot.SignalGenerator.format_signal_text", return_value=self.expected_signal_text), \
                 patch("run_copilot.gemini_analyst.analyze_setup", new=AsyncMock(side_effect=ConnectionError("Simulated socket reset"))), \
                 self.assertLogs("TradiCopilot", level="WARNING") as log_cm:
                
                price, dispatched = await run_copilot.scan_single_symbol(
                    "BTCUSDT", self.mock_client, self.mock_smc, self.mock_renderer, mock_telegram
                )
                return price, dispatched, log_cm.output

        price, dispatched, logs = asyncio.run(run_test())

        self.assertEqual(len(dispatched), 1)
        final_text = dispatched[0][1]
        self.assertEqual(final_text, self.expected_signal_text)
        self.assertTrue(any("Error al analizar setup con Gemini para BTCUSDT" in msg for msg in logs))
        mock_telegram.send_signal.assert_called_once_with(
            self.expected_signal_text, "output/charts/test_btcusdt.png", setup=self.setup
        )

    def test_case_c_timeout_error_fallback(self):
        """Case C variation: Gemini raises asyncio.TimeoutError -> safe fallback."""
        mock_telegram = MagicMock(spec=TelegramNotifier)
        mock_telegram.is_configured.return_value = True

        async def run_test():
            with patch("run_copilot.SignalGenerator.format_signal_text", return_value=self.expected_signal_text), \
                 patch("run_copilot.gemini_analyst.analyze_setup", new=AsyncMock(side_effect=asyncio.TimeoutError("Timeout"))):
                
                return await run_copilot.scan_single_symbol(
                    "BTCUSDT", self.mock_client, self.mock_smc, self.mock_renderer, mock_telegram
                )

        _, dispatched = asyncio.run(run_test())
        self.assertEqual(dispatched[0][1], self.expected_signal_text)

    def test_case_c_quota_or_unexpected_exception_fallback(self):
        """Case C variation: Gemini raises unexpected Exception (e.g. 429 ResourceExhausted)."""
        mock_telegram = MagicMock(spec=TelegramNotifier)
        mock_telegram.is_configured.return_value = True

        async def run_test():
            with patch("run_copilot.SignalGenerator.format_signal_text", return_value=self.expected_signal_text), \
                 patch("run_copilot.gemini_analyst.analyze_setup", new=AsyncMock(side_effect=Exception("429 Resource Exhausted"))):
                
                return await run_copilot.scan_single_symbol(
                    "BTCUSDT", self.mock_client, self.mock_smc, self.mock_renderer, mock_telegram
                )

        _, dispatched = asyncio.run(run_test())
        self.assertEqual(dispatched[0][1], self.expected_signal_text)

    # =========================================================================
    # Group D: Case D — 1-Click buttons keyboard object continuity
    # =========================================================================

    def test_case_d_send_signal_keyboard_generation_intact(self):
        """
        Case D1: Verify real TelegramNotifier.send_signal builds the 1-Click keyboard from setup.
        Assert inline keyboard buttons:
        - Button 1: text contains '🟢 EJECUTAR EN BITUNIX', callback_data contains 'exec:BTCUSDT:LONG:64000.0:63000.0:66500.0'
        - Button 2: text contains '❌ DESCARTAR SEÑAL', callback_data == 'discard'
        """
        real_notifier = TelegramNotifier(token="fake_token", chat_id="fake_chat_id")

        with patch.object(real_notifier, "send_message") as mock_send_msg, \
             patch("run_copilot.SignalGenerator.format_signal_text", return_value=self.expected_signal_text), \
             patch("run_copilot.gemini_analyst.analyze_setup", new=AsyncMock(return_value="Valid AI")):

            # Point chart to non-existent file so send_message is invoked with reply_markup
            self.mock_renderer.render_trade_setup.return_value = "non_existent_chart.png"

            asyncio.run(
                run_copilot.scan_single_symbol("BTCUSDT", self.mock_client, self.mock_smc, self.mock_renderer, real_notifier)
            )

            mock_send_msg.assert_called_once()
            called_text = mock_send_msg.call_args[0][0]
            called_markup = mock_send_msg.call_args[1].get("reply_markup") or mock_send_msg.call_args[0][1]

            self.assertIn("🤖 ANÁLISIS IA — BTCUSDT", called_text)
            self.assertIsNotNone(called_markup)
            self.assertIn("inline_keyboard", called_markup)

            rows = called_markup["inline_keyboard"]
            self.assertEqual(len(rows), 2)
            # Button 1: Execute
            self.assertIn("🟢 EJECUTAR EN BITUNIX", rows[0][0]["text"])
            self.assertEqual(rows[0][0]["callback_data"], "exec:BTCUSDT:LONG:64000.0:63000.0:66500.0")
            # Button 2: Discard
            self.assertEqual(rows[1][0]["text"], "❌ DESCARTAR SEÑAL")
            self.assertEqual(rows[1][0]["callback_data"], "discard")

    def test_case_d_notify_signal_coroutine_receives_setup_intact(self):
        """
        Case D2: Verify that when telegram implements notify_signal (async coroutine),
        setup=s is passed intact, enabling keyboard construction by downstream handler.
        """
        class MockTelegramWithNotifyAsync:
            def __init__(self):
                self.calls = []

            def is_configured(self):
                return True

            async def notify_signal(self, setup, chart_bytes, signal_text):
                # Verify downstream can build trade keyboard from setup
                keyboard = TelegramNotifier._build_trade_keyboard(None, setup=setup)
                self.calls.append({
                    "setup": setup,
                    "chart_bytes": chart_bytes,
                    "signal_text": signal_text,
                    "keyboard": keyboard
                })

        mock_tel = MockTelegramWithNotifyAsync()

        async def run_test():
            with patch("run_copilot.SignalGenerator.format_signal_text", return_value=self.expected_signal_text), \
                 patch("run_copilot.gemini_analyst.analyze_setup", new=AsyncMock(return_value="AI analysis text")):
                
                return await run_copilot.scan_single_symbol(
                    "BTCUSDT", self.mock_client, self.mock_smc, self.mock_renderer, mock_tel
                )

        asyncio.run(run_test())

        self.assertEqual(len(mock_tel.calls), 1)
        record = mock_tel.calls[0]
        self.assertIs(record["setup"], self.setup)
        self.assertIn("🤖 ANÁLISIS IA — BTCUSDT", record["signal_text"])
        self.assertEqual(record["keyboard"]["inline_keyboard"][0][0]["callback_data"],
                         "exec:BTCUSDT:LONG:64000.0:63000.0:66500.0")

    def test_case_d_notify_signal_sync_receives_setup_intact(self):
        """
        Case D3: Verify that when telegram implements notify_signal (synchronous function),
        asyncio.to_thread runs it and setup=s is passed intact.
        """
        class MockTelegramWithNotifySync:
            def __init__(self):
                self.calls = []

            def is_configured(self):
                return True

            def notify_signal(self, setup, chart_bytes, signal_text):
                keyboard = TelegramNotifier._build_trade_keyboard(None, setup=setup)
                self.calls.append({
                    "setup": setup,
                    "chart_bytes": chart_bytes,
                    "signal_text": signal_text,
                    "keyboard": keyboard
                })

        mock_tel = MockTelegramWithNotifySync()

        async def run_test():
            with patch("run_copilot.SignalGenerator.format_signal_text", return_value=self.expected_signal_text), \
                 patch("run_copilot.gemini_analyst.analyze_setup", new=AsyncMock(return_value="Sync AI test")):
                
                return await run_copilot.scan_single_symbol(
                    "BTCUSDT", self.mock_client, self.mock_smc, self.mock_renderer, mock_tel
                )

        asyncio.run(run_test())

        self.assertEqual(len(mock_tel.calls), 1)
        record = mock_tel.calls[0]
        self.assertEqual(record["setup"].symbol, "BTCUSDT")
        self.assertIn("Sync AI test", record["signal_text"])
        self.assertEqual(record["keyboard"]["inline_keyboard"][1][0]["callback_data"], "discard")

    def test_case_d_keyboard_64_byte_truncation_rule(self):
        """
        Case D4: Stress test callback_data > 64 bytes.
        Verifies TelegramNotifier._build_trade_keyboard switches to compact 'ex:' format
        so Telegram does not reject the inline keyboard.
        """
        long_setup = TradeSetup(
            symbol="VERYLONGSYMBOLNAME123USDT",
            direction="LONG",
            entry_price=123456.78901234,
            stop_loss=120000.12345678,
            tp1=125000.0,
            tp2=128000.98765432,
            tp3=130000.0,
            risk_distance=3456.0,
            rr_tp1=1.5,
            rr_tp2=2.5,
            rr_tp3=4.0,
            confidence_score=90.0,
            reasons=["OB"]
        )

        notifier = TelegramNotifier(token="fake", chat_id="fake")
        keyboard = notifier._build_trade_keyboard(long_setup)
        cb_data = keyboard["inline_keyboard"][0][0]["callback_data"]

        self.assertLessEqual(len(cb_data.encode("utf-8")), 64, f"Callback data superó 64 bytes: {cb_data}")
        self.assertTrue(cb_data.startswith("ex:") or cb_data.startswith("exec:"))

    # =========================================================================
    # Group E: Market Scanner Edge Cases
    # =========================================================================

    def test_case_e_unconfigured_telegram_does_not_call_send_or_crash(self):
        """Case E1: When Telegram is not configured, signal is created but no send is attempted."""
        mock_telegram = MagicMock(spec=TelegramNotifier)
        mock_telegram.is_configured.return_value = False

        async def run_test():
            with patch("run_copilot.SignalGenerator.format_signal_text", return_value=self.expected_signal_text), \
                 patch("run_copilot.gemini_analyst.analyze_setup", new=AsyncMock(return_value="Analysis")):
                
                return await run_copilot.scan_single_symbol(
                    "BTCUSDT", self.mock_client, self.mock_smc, self.mock_renderer, mock_telegram
                )

        price, dispatched = asyncio.run(run_test())
        self.assertEqual(len(dispatched), 1)
        mock_telegram.send_signal.assert_not_called()
        self.assertIn("🤖 ANÁLISIS IA", dispatched[0][1])

    def test_case_e_insufficient_klines_returns_none(self):
        """Case E2: MTF klines dataframe with fewer than 30 rows returns None, None."""
        short_df = create_dummy_df(20)
        self.mock_client.get_historical_klines.side_effect = None
        self.mock_client.get_historical_klines.return_value = short_df
        mock_telegram = MagicMock(spec=TelegramNotifier)

        price, dispatched = asyncio.run(
            run_copilot.scan_single_symbol("BTCUSDT", self.mock_client, self.mock_smc, self.mock_renderer, mock_telegram)
        )
        self.assertIsNone(price)
        self.assertIsNone(dispatched)

    def test_case_e_multiple_setups_dispatched_individually(self):
        """Case E3: Multiple setups detected in a single symbol are all analyzed and dispatched."""
        setup1 = create_dummy_setup("ETHUSDT")
        setup1.direction = "LONG"
        setup2 = create_dummy_setup("ETHUSDT")
        setup2.direction = "SHORT"
        self.mock_smc.analyze.return_value = {"setups": [setup1, setup2]}

        mock_telegram = MagicMock(spec=TelegramNotifier)
        mock_telegram.is_configured.return_value = True

        async def run_test():
            with patch("run_copilot.SignalGenerator.format_signal_text", return_value="Signal formatted"), \
                 patch("run_copilot.gemini_analyst.analyze_setup", new=AsyncMock(side_effect=["AI Analysis 1", "AI Analysis 2"])):
                
                return await run_copilot.scan_single_symbol(
                    "ETHUSDT", self.mock_client, self.mock_smc, self.mock_renderer, mock_telegram
                )

        _, dispatched = asyncio.run(run_test())
        self.assertEqual(len(dispatched), 2)
        self.assertIn('"AI Analysis 1"', dispatched[0][1])
        self.assertIn('"AI Analysis 2"', dispatched[1][1])
        self.assertEqual(mock_telegram.send_signal.call_count, 2)


if __name__ == "__main__":
    unittest.main()

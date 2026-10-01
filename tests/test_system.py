import unittest
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import settings
from src.patterns.types import TradeSetup, FVG, OrderBlock, LiquiditySweep
from src.patterns.fvg import detect_fvgs
from src.patterns.order_blocks import detect_order_blocks
from src.patterns.liquidity import detect_liquidity_sweeps
from src.patterns.pivots import detect_swing_points
from src.patterns.smc_engine import SMCEngine
from src.intelligence.features import extract_features, FEATURE_NAMES, calculate_rsi, calculate_adx
from src.intelligence.ml_model import AdaptiveTradingBrain
from src.exchanges.bitunix.trader import BitunixTrader
from src.simulation.monte_carlo import run_monte_carlo_simulation, calculate_time_to_milestones
from src.copilot.signal_generator import SignalGenerator
from src.copilot.chart_renderer import ChartRenderer

class TestSystemQuantitativeSuite(unittest.TestCase):
    """Suite de pruebas unitarias e integrales para el sistema cuantitativo Tradi."""

    def setUp(self):
        np.random.seed(42)
        n = 100
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.4) + 0.1
        low = close - np.abs(np.random.randn(n) * 0.4) - 0.1
        open_p = close + np.random.randn(n) * 0.2
        volume = np.random.lognormal(mean=7, sigma=0.5, size=n)
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="15min")

        self.df_sample = pd.DataFrame({
            "timestamp": dates,
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume
        })

    def test_feature_extraction_completeness_and_shapes(self):
        """Verifica que se extraigan exactamente las 24 características continuas sin NaNs ni Infs."""
        feats = extract_features(self.df_sample, last_n=None)
        self.assertEqual(len(feats.columns), len(FEATURE_NAMES))
        for col in FEATURE_NAMES:
            self.assertIn(col, feats.columns)
            self.assertFalse(feats[col].isna().any(), f"NaN detectado en {col}")
            self.assertFalse(np.isinf(feats[col]).any(), f"Inf detectado en {col}")

        # Comprobar tail last_n
        single_row = extract_features(self.df_sample, last_n=1)
        self.assertEqual(len(single_row), 1)
        self.assertEqual(len(single_row.columns), 24)

    def test_feature_extraction_htf_timezone_resilience(self):
        """Verifica que extract_features no falle si df y df_htf tienen zonas horarias distintas o sin zona."""
        df_naive = self.df_sample.copy()
        df_naive["timestamp"] = df_naive["timestamp"].dt.tz_localize(None)

        df_htf = pd.DataFrame({
            "timestamp": pd.date_range(end=datetime.now(timezone.utc), periods=30, freq="4h"),
            "open": np.linspace(95, 105, 30),
            "high": np.linspace(96, 106, 30),
            "low": np.linspace(94, 104, 30),
            "close": np.linspace(95, 105, 30),
            "volume": np.full(30, 1000)
        })

        # Merge entre naive y aware
        feats = extract_features(df_naive, df_htf=df_htf, last_n=10)
        self.assertEqual(len(feats), 10)
        self.assertIn("feat_htf_trend_align", feats.columns)
        self.assertFalse(feats["feat_htf_trend_align"].isna().any())

    def test_indicators_rsi_and_adx(self):
        """Verifica el cálculo acotado de RSI (0 a 100) y ADX."""
        rsi = calculate_rsi(self.df_sample["close"], period=14)
        self.assertTrue((rsi >= 0.0).all() and (rsi <= 100.0).all())

        adx = calculate_adx(self.df_sample, period=14)
        self.assertTrue((adx >= 0.0).all())
        self.assertFalse(adx.isna().any())

    def test_pattern_detection_fvg_and_order_blocks(self):
        """Verifica la detección de estructuras SMC (FVG y Order Blocks)."""
        smc = SMCEngine(atr_period=14)
        atr = smc.calculate_atr(self.df_sample)
        fvgs = detect_fvgs(self.df_sample, atr)
        self.assertIsInstance(fvgs, list)

        obs = detect_order_blocks(self.df_sample, fvgs, atr)
        self.assertIsInstance(obs, list)

        sh, sl = detect_swing_points(self.df_sample, left=3, right=3)
        sweeps = detect_liquidity_sweeps(self.df_sample, sh, sl)
        self.assertIsInstance(sweeps, list)

    def test_ml_model_prediction_and_bounds(self):
        """Verifica que AdaptiveTradingBrain cargue el modelo entrenado y compute P(Win) acotada."""
        brain = AdaptiveTradingBrain()
        self.assertIsNotNone(brain.ml_pipeline, "El pipeline de ML debe estar cargado desde data/tradi_ml_model.joblib")

        setup = TradeSetup(
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=60000.0,
            stop_loss=59200.0,
            tp1=61200.0,
            tp2=62400.0,
            tp3=64000.0,
            risk_distance=800.0,
            rr_tp1=1.5,
            rr_tp2=3.0,
            rr_tp3=5.0,
            confidence_score=80.0,
            reasons=["Rebote confirmado en Order Block Alcista", "Barrido de liquidez SSL Sweep"]
        )

        feat_row = extract_features(self.df_sample, last_n=1).iloc[0]
        prob = brain.predict_probability(setup, feat_row)
        
        self.assertIsInstance(prob, float)
        self.assertTrue(25.0 <= prob <= 96.0, f"Probabilidad fuera de rango: {prob}")

    def test_bitunix_trader_dynamic_kelly_and_hard_leverage_cap(self):
        """Verifica que BitunixTrader aplique Dynamic Kelly (2.0%) y respete estrictamente los límites de apalancamiento y margen."""
        trader = BitunixTrader(api_key="", api_secret="", account_equity=500.0, default_risk_pct=0.02)
        self.assertEqual(trader.max_leverage_notional, settings.MAX_LEVERAGE_NOTIONAL)

        # Caso 1: SL muy estrecho que intentaría un apalancamiento excesivo (> 2.5x)
        # Entry: $100, SL: $99.9 (Distancia $0.1). Riesgo deseado 2% de $500 = $10 USD.
        # Hard cap 2.5x en $500 = $1,250 USD max notional -> Qty máxima permitida <= 12.5 unidades.
        qty_capped = trader.calculate_position_size("SOLUSDT", entry_price=100.0, stop_loss=99.9)
        notional_capped = qty_capped * 100.0
        self.assertLessEqual(notional_capped, 500.0 * trader.max_leverage_notional * 1.01, f"Hard Leverage Cap superado: Notional ${notional_capped}")

        # Caso 2: Crecimiento de cuenta a $2,500 USD
        trader.update_account_equity(2500.0)
        self.assertEqual(trader.account_equity, 2500.0)
        # 2% de $2,500 = $50 USD riesgo
        qty_compounded = trader.calculate_position_size("SOLUSDT", entry_price=100.0, stop_loss=95.0)
        # Distance $5 -> Qty = 50 / 5 = 10 unidades -> Notional $1,000 USD (bien dentro de los límites).
        self.assertLessEqual(qty_compounded, 10.0)

        # Caso 3: Ejecución de orden virtual segura
        exec_res = trader.execute_order(
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=60000.0,
            stop_loss=59000.0,
            take_profit=63000.0
        )
        self.assertTrue(exec_res["success"])
        self.assertEqual(exec_res["mode"], "SIMULATION")
        self.assertIn("estimated_margin", exec_res)
        self.assertLessEqual(exec_res["leverage"], trader.max_leverage_notional)

    def test_multi_asset_universe_and_position_sizing(self):
        """Verifica que el universo multi-activo contenga commodities y acciones, y que el dimensionamiento de posición sea exacto y respete el margen."""
        required_assets = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XAUUSDT", "XAGUSDT", "CLUSDT", "SPCXUSDT", "NVDAUSDT", "TSLAUSDT"]
        for asset in required_assets:
            self.assertIn(asset, settings.DEFAULT_SYMBOLS, f"{asset} debe estar en DEFAULT_SYMBOLS")

        trader_50 = BitunixTrader(api_key="", api_secret="", account_equity=50.0, default_risk_pct=0.02)
        
        # Test Gold (XAUUSDT @ $4,000, SL $3,980 -> dist $20, risk $1.0 -> qty capped by margin)
        qty_gold = trader_50.calculate_position_size("XAUUSDT", entry_price=4000.0, stop_loss=3980.0)
        self.assertGreater(qty_gold, 0.0)
        self.assertLessEqual(qty_gold * 4000.0, 50.0 * settings.MAX_LEVERAGE_NOTIONAL * 1.01)

        # Test Oil (CLUSDT @ $90.0, SL $89.0 -> dist $1.0 -> max margin 8% = $4.0 USD -> notional $40 -> qty 0.4 CL con base_prec 1)
        qty_oil = trader_50.calculate_position_size("CLUSDT", entry_price=90.0, stop_loss=89.0)
        self.assertEqual(qty_oil, 0.4)

        # Test BNB (BNBUSDT @ $600.0, SL $580.0 -> dist $20.0 -> risk $1.0 -> qty 0.05 BNB con base_prec 2)
        qty_bnb = trader_50.calculate_position_size("BNBUSDT", entry_price=600.0, stop_loss=580.0)
        self.assertEqual(qty_bnb, 0.05)
        self.assertGreaterEqual(qty_bnb, 0.01)

        # Test Nvidia (NVDAUSDT @ $230.0, SL $225.0 -> dist $5.0 -> qty bounded by margin)
        qty_nvda = trader_50.calculate_position_size("NVDAUSDT", entry_price=230.0, stop_loss=225.0)
        self.assertGreater(qty_nvda, 0.0)
        self.assertLessEqual(qty_nvda * 230.0, 50.0 * settings.MAX_LEVERAGE_NOTIONAL * 1.01)

    def test_monte_carlo_and_time_to_milestones(self):
        """Verifica la simulación Monte Carlo Bootstrap y las estimaciones temporales a hitos ($1k, $2.5k, $5k, $10k, $25k)."""
        mc = run_monte_carlo_simulation(
            win_rate=0.78,
            reward_risk=2.6,
            risk_per_trade_pct=0.02,
            n_simulations=10000,
            n_trades=200,
            initial_capital=500.0,
            trades_per_week=6.0
        )

        self.assertIn("milestones_summary", mc)
        self.assertIn("prob_of_ruin", mc)
        self.assertEqual(mc["prob_of_ruin"], 0.0, "La probabilidad de ruina debe ser 0.00% con Kelly 2% y WR > 70%")
        self.assertGreater(mc["expected_final_equity"], 500.0)

        milestones = mc["milestones_summary"]
        expected_targets = ["$1,000 USD", "$2,500 USD", "$5,000 USD", "$10,000 USD", "$25,000 USD"]
        for target in expected_targets:
            self.assertIn(target, milestones)
            m_data = milestones[target]
            self.assertGreaterEqual(m_data["prob_reached"], 0.90)
            self.assertIsNotNone(m_data["p50_trades_median"])
            self.assertIsNotNone(m_data["p50_months_median"])
            self.assertGreater(m_data["p50_months_median"], 0.0)

    def test_copilot_signal_formatting_and_chart_generation(self):
        """Verifica el formato del mensaje institucional y la renderización gráfica del setup."""
        setup = TradeSetup(
            symbol="ETHUSDT",
            direction="LONG",
            entry_price=3000.0,
            stop_loss=2950.0,
            tp1=3075.0,
            tp2=3150.0,
            tp3=3250.0,
            risk_distance=50.0,
            rr_tp1=1.5,
            rr_tp2=3.0,
            rr_tp3=5.0,
            confidence_score=88.5,
            reasons=["Mitigación de FVG Alcista", "Alineación Macro 4H Bullish"]
        )

        mc = run_monte_carlo_simulation(win_rate=0.88, reward_risk=3.0, n_simulations=1000)
        formatted_signal = SignalGenerator.format_signal_text(setup, mc)

        self.assertIn("ETHUSDT", formatted_signal)
        self.assertIn("LONG (COMPRA)", formatted_signal)
        self.assertIn("$3,000.00", formatted_signal)
        self.assertIn("Probabilidad de Ruina:", formatted_signal)
        self.assertIn("Max Drawdown Esperado", formatted_signal)

        renderer = ChartRenderer()
        chart_path = renderer.render_trade_setup(self.df_sample, setup)
        self.assertTrue(os.path.exists(chart_path), f"El archivo del gráfico no se creó: {chart_path}")

if __name__ == "__main__":
    unittest.main()

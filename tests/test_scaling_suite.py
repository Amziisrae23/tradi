import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import tempfile
import pytest
from src.storage.trading_ledger import TradingLedger
from src.intelligence.portfolio_risk import PortfolioRiskManager
from src.copilot.telegram_notifier import TelegramNotifier


class TestScalingSuite:

    @pytest.fixture
    def temp_ledger(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_path = tmp.name
        ledger_inst = TradingLedger(db_path=tmp_path)
        yield ledger_inst
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

    def test_trading_ledger_full_lifecycle(self, temp_ledger):
        # 1. Registrar apertura de 3 trades
        assert temp_ledger.record_trade_opened(
            order_id="TEST-001",
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=85000.0,
            stop_loss=84000.0,
            tp1=86500.0,
            tp2=87500.0,
            tp3=90000.0,
            risk_usd=1.0,
            gemini_verdict="EJECUTAR"
        ) is True

        assert temp_ledger.record_trade_opened(
            order_id="TEST-002",
            symbol="ETHUSDT",
            direction="SHORT",
            entry_price=2700.0,
            stop_loss=2730.0,
            tp1=2655.0,
            tp2=2620.0,
            tp3=2550.0,
            risk_usd=1.0,
            gemini_verdict="EJECUTAR"
        ) is True

        # 2. Registrar cierres: 1 ganador (+1.8 USD), 1 perdedor (-1.0 USD)
        assert temp_ledger.record_trade_closed(
            order_id="TEST-001",
            exit_price=87500.0,
            pnl_usd=1.80,
            pnl_r=1.8,
            status="TP2_TARGET"
        ) is True

        assert temp_ledger.record_trade_closed(
            order_id="TEST-002",
            exit_price=2730.0,
            pnl_usd=-1.00,
            pnl_r=-1.0,
            status="STOP_LOSS"
        ) is True

        # 3. Validar métricas estadísticas
        stats = temp_ledger.get_stats()
        assert stats["total_trades"] == 2
        assert stats["wins"] == 1
        assert stats["losses"] == 1
        assert stats["win_rate"] == 50.0
        assert stats["total_pnl_usd"] == 0.80
        assert stats["profit_factor"] == 1.80

    def test_portfolio_risk_manager_limits(self):
        risk_mgr = PortfolioRiskManager(max_concurrent_trades=2, daily_loss_limit_pct=0.05)

        # 1. Sin trades activos -> debe aprobar
        approved, reason = risk_mgr.can_open_trade("BTCUSDT", "LONG", [], account_equity=50.0)
        assert approved is True

        # 2. Con 2 trades activos -> debe bloquear por calor de cartera
        active_sample = [
            {"symbol": "ETHUSDT", "status": "OPEN"},
            {"symbol": "SOLUSDT", "status": "OPEN"}
        ]
        approved, reason = risk_mgr.can_open_trade("BTCUSDT", "LONG", active_sample, account_equity=50.0)
        assert approved is False
        assert "Calor de cartera al límite" in reason

        # 3. Activo duplicado -> debe bloquear
        active_one = [{"symbol": "BTCUSDT", "status": "OPEN"}]
        approved, reason = risk_mgr.can_open_trade("BTCUSDT", "SHORT", active_one, account_equity=50.0)
        assert approved is False
        assert "Ya existe una posición abierta" in reason

        # 4. Modo Pausa manual -> debe bloquear
        risk_mgr.is_paused = True
        approved, reason = risk_mgr.can_open_trade("XRPUSDT", "LONG", [], account_equity=50.0)
        assert approved is False
        assert "modo PAUSA" in reason

    def test_position_monitor_live_reconciliation_and_close(self):
        from src.copilot.position_monitor import PositionMonitor
        from src.exchanges.bitunix.trader import BitunixTrader

        mon = PositionMonitor()
        mon.active_trades = []

        # Registrar trade virtual
        mon.register_trade(
            order_id="TEST-SIM-CLOSE",
            symbol="LINKUSDT",
            direction="LONG",
            entry_price=14.0,
            stop_loss=13.5,
            tp1=14.5,
            tp2=15.0,
            tp3=16.0,
            risk_usd=1.0
        )
        assert len(mon.get_active_trades()) == 1

        # Simular gatillo de SL: price = 13.4 <= 13.5
        mon.check_market_prices({"LINKUSDT": 13.4})
        assert len(mon.get_active_trades()) == 0

        # Probar trader close methods en simulación
        trader_sim = BitunixTrader(api_key="", api_secret="")
        close_res = trader_sim.close_position("POS-123", "LINKUSDT")
        assert close_res["success"] is True
        assert close_res["mode"] == "SIMULATION"

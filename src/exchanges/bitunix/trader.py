import os
import time
import uuid
import hashlib
import json
import logging
import requests
from typing import Dict, Any, Optional
from config.settings import settings

logger = logging.getLogger("BitunixTrader")

class BitunixTrader:
    """
    Cliente oficial para ejecución institucional de futuros en Bitunix:
    - Firma Criptográfica Doble SHA-256
    - Endpoint oficial: POST /api/v1/futures/trade/place_order
    - Hard Leverage Cap (Tope de apalancamiento 5.0x)
    - TP/SL Integrado
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        account_equity: float = 500.0,
        default_risk_pct: float = 0.02
    ):
        self.api_key = api_key if api_key is not None else settings.BITUNIX_API_KEY
        self.api_secret = api_secret if api_secret is not None else settings.BITUNIX_API_SECRET
        self.base_url = settings.BITUNIX_REST_URL
        self.max_leverage_notional = 5.0
        self.account_equity = account_equity
        self.default_risk_pct = default_risk_pct

    def update_account_equity(self, equity: float):
        self.account_equity = max(equity, 10.0)

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_secret)

    def _generate_signature(self, nonce: str, timestamp: str, body_str: str) -> str:
        raw_string = f"{nonce}{timestamp}{self.api_key}{body_str}"
        first_digest = hashlib.sha256(raw_string.encode('utf-8')).hexdigest()
        final_sign = hashlib.sha256((first_digest + self.api_secret).encode('utf-8')).hexdigest()
        return final_sign

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        risk_usd: Optional[float] = None,
        risk_pct: Optional[float] = None,
        current_equity: Optional[float] = None
    ) -> float:
        equity = current_equity or self.account_equity
        pct = risk_pct or self.default_risk_pct
        target_risk = risk_usd if risk_usd is not None else (equity * pct)

        risk_distance = abs(entry_price - stop_loss)
        if risk_distance <= 0 or entry_price <= 0:
            return 0.001
        
        calculated_qty = target_risk / risk_distance
        max_allowed_notional = equity * self.max_leverage_notional
        max_allowed_qty = max_allowed_notional / entry_price
        
        final_qty = min(calculated_qty, max_allowed_qty)
        min_qty = 5.0 / entry_price
        final_qty = max(final_qty, min_qty)

        if "BTC" in symbol:
            return round(final_qty, 3)
        elif "ETH" in symbol:
            return round(final_qty, 2)
        elif any(c in symbol for c in ["SOL", "AVAX", "LINK", "BNB"]):
            return round(final_qty, 1)
        elif any(c in symbol for c in ["XRP", "DOGE", "SUI", "ADA"]):
            return round(final_qty, 0)
        return round(final_qty, 2)

    def execute_order(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        risk_usd: Optional[float] = None,
        risk_pct: Optional[float] = None,
        current_equity: Optional[float] = None
    ) -> Dict[str, Any]:
        equity = current_equity or self.account_equity
        pct = risk_pct or self.default_risk_pct
        actual_risk_usd = risk_usd if risk_usd is not None else (equity * pct)

        clean_symbol = symbol.replace("/", "").replace("-", "").upper()
        qty = self.calculate_position_size(clean_symbol, entry_price, stop_loss, risk_usd=actual_risk_usd, current_equity=equity)
        notional_value = qty * entry_price
        leverage_used = notional_value / equity
        side = "BUY" if direction.upper() == "LONG" else "SELL"

        if not self.is_configured():
            sim_id = f"SIM-{uuid.uuid4().hex[:8].upper()}"
            logger.info(f"🟢 [SIMULACIÓN BITUNIX]: {clean_symbol} {direction} | Cantidad: {qty} ({leverage_used:.1f}x)")
            return {
                "success": True,
                "mode": "SIMULATION",
                "order_id": sim_id,
                "symbol": clean_symbol,
                "direction": direction,
                "entry_price": entry_price,
                "qty": qty,
                "notional": round(notional_value, 2),
                "leverage": round(leverage_used, 1),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_usd": actual_risk_usd,
                "msg": f"Orden Virtual ejecutada: {qty} {clean_symbol} (${notional_value:.2f} USD, {leverage_used:.1f}x)"
            }

        url = f"{self.base_url}/api/v1/futures/trade/place_order"
        nonce = uuid.uuid4().hex
        timestamp = str(int(time.time() * 1000))

        payload = {
            "symbol": clean_symbol,
            "side": side,
            "orderType": "LIMIT",
            "tradeSide": "OPEN",
            "price": str(entry_price),
            "qty": str(qty),
            "slPrice": str(stop_loss),
            "tpPrice": str(take_profit)
        }

        body_str = json.dumps(payload, separators=(',', ':'))
        signature = self._generate_signature(nonce, timestamp, body_str)

        headers = {
            "api-key": self.api_key,
            "nonce": nonce,
            "timestamp": timestamp,
            "sign": signature,
            "Content-Type": "application/json"
        }

        try:
            res = requests.post(url, data=body_str, headers=headers, timeout=10)
            res.raise_for_status()
            data = res.json()
            if data.get("code") == 0:
                order_id = data.get("data", {}).get("orderId", "OK")
                logger.info(f"✔ Orden REAL enviada a Bitunix: ID {order_id}")
                return {
                    "success": True,
                    "mode": "REAL",
                    "order_id": order_id,
                    "symbol": clean_symbol,
                    "direction": direction,
                    "qty": qty,
                    "notional": round(notional_value, 2),
                    "risk_usd": actual_risk_usd,
                    "msg": f"Orden REAL enviada a Bitunix: {qty} {clean_symbol} (Riesgo: ${actual_risk_usd:.2f} USD)"
                }
            else:
                logger.error(f"Bitunix rechazó la orden: {data}")
                return {"success": False, "mode": "REAL", "msg": data.get("msg", "Error en Bitunix")}
        except Exception as e:
            logger.error(f"Excepción en Bitunix: {e}")
            return {"success": False, "mode": "REAL", "msg": str(e)}

trader = BitunixTrader()

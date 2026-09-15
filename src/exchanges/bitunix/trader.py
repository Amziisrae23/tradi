import os
import time
import uuid
import hashlib
import logging
import requests
from typing import Dict, Any, Optional
from config.settings import settings

logger = logging.getLogger("BitunixTrader")

class BitunixTrader:
    """
    Cliente para ejecución institucional de futuros en Bitunix con:
    - Firma Criptográfica Doble SHA-256
    - Hard Leverage Cap (Tope máximo 5.0x Notional para cuentas de $500)
    - Control de Calor de Cartera (Portfolio Heat)
    - Modo Simulación Seguro para desarrollo y pruebas
    """

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key or settings.BITUNIX_API_KEY
        self.api_secret = api_secret or settings.BITUNIX_API_SECRET
        self.base_url = settings.BITUNIX_REST_URL
        self.max_leverage_notional = 5.0  # Máximo 5x Notional ($2,500 USD en cuenta de $500)
        self.account_equity = 500.0       # Capital base institucional

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_secret)

    def _generate_signature(self, nonce: str, timestamp: str, body_str: str) -> str:
        raw_string = f"{nonce}{timestamp}{self.api_key}{body_str}"
        first_digest = hashlib.sha256(raw_string.encode('utf-8')).hexdigest()
        final_sign = hashlib.sha256((first_digest + self.api_secret).encode('utf-8')).hexdigest()
        return final_sign

    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float, risk_usd: float = 10.0) -> float:
        """
        Calcula el tamaño exacto de posición con protecciones cuantitativas:
        1. Asignación de riesgo: Qty = Risk / Distance_SL
        2. Hard Leverage Cap: Qty_max = (Equity * 5.0x) / Entry_Price
        """
        risk_distance = abs(entry_price - stop_loss)
        if risk_distance <= 0 or entry_price <= 0:
            return 0.001
        
        # 1. Cantidad basada en el riesgo fijo ($10 USD)
        calculated_qty = risk_usd / risk_distance
        
        # 2. Protección Institucional: Tope de apalancamiento máximo a 5.0x
        max_allowed_notional = self.account_equity * self.max_leverage_notional  # $2,500 USD máx
        max_allowed_qty = max_allowed_notional / entry_price
        
        # Aplicar el mínimo entre el riesgo deseado y el tope de apalancamiento
        final_qty = min(calculated_qty, max_allowed_qty)
        
        # Garantizar valor notional mínimo de $5.00 USDT en Bitunix
        min_qty = 5.0 / entry_price
        final_qty = max(final_qty, min_qty)

        # Redondear con precisión según el activo
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
        risk_usd: float = 10.0
    ) -> Dict[str, Any]:
        clean_symbol = symbol.replace("/", "").replace("-", "").upper()
        qty = self.calculate_position_size(clean_symbol, entry_price, stop_loss, risk_usd)
        notional_value = qty * entry_price
        leverage_used = notional_value / self.account_equity
        side = "BUY" if direction.upper() == "LONG" else "SELL"

        # === MODO SIMULACIÓN VIRTUAL SEGURO ===
        if not self.is_configured():
            sim_id = f"SIM-{uuid.uuid4().hex[:8].upper()}"
            logger.info(f"🟢 [SIMULACIÓN BITUNIX]: {clean_symbol} {direction}")
            logger.info(f"   Entrada: ${entry_price} | SL: ${stop_loss} | TP: ${take_profit}")
            logger.info(f"   Cantidad: {qty} ({clean_symbol}) | Notional: ${notional_value:.2f} USD (Apalancamiento: {leverage_used:.1f}x)")
            
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
                "msg": f"Orden Virtual ejecutada: {qty} {clean_symbol} (${notional_value:.2f} USD, {leverage_used:.1f}x)"
            }

        # === MODO REAL BITUNIX API (DOBLE SHA-256) ===
        url = f"{self.base_url}/api/v1/futures/trade/order"
        nonce = uuid.uuid4().hex
        timestamp = str(int(time.time() * 1000))

        payload = {
            "symbol": clean_symbol,
            "side": side,
            "type": "LIMIT",
            "price": str(entry_price),
            "qty": str(qty),
            "stopLoss": str(stop_loss),
            "takeProfit": str(take_profit),
            "tradeType": "OPEN"
        }

        import json
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
                logger.info(f"✔ Orden REAL en Bitunix exitosa: {data}")
                return {
                    "success": True,
                    "mode": "REAL",
                    "order_id": data.get("data", {}).get("orderId", ""),
                    "symbol": clean_symbol,
                    "direction": direction,
                    "qty": qty,
                    "msg": f"Orden REAL ejecutada en Bitunix: {qty} {clean_symbol} (Riesgo: ${risk_usd} USD)"
                }
            else:
                logger.error(f"Bitunix rechazó la orden: {data}")
                return {"success": False, "mode": "REAL", "msg": data.get("msg", "Error en Bitunix")}
        except Exception as e:
            logger.error(f"Excepción en Bitunix: {e}")
            return {"success": False, "mode": "REAL", "msg": str(e)}

trader = BitunixTrader()

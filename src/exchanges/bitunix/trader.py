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
    """Cliente para ejecución de órdenes de futuros en Bitunix con firma Doble SHA-256."""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key or settings.BITUNIX_API_KEY
        self.api_secret = api_secret or settings.BITUNIX_API_SECRET
        self.base_url = settings.BITUNIX_REST_URL

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_secret)

    def _generate_signature(self, nonce: str, timestamp: str, body_str: str) -> str:
        """Algoritmo de firma Doble SHA-256 oficial de Bitunix."""
        raw_string = f"{nonce}{timestamp}{self.api_key}{body_str}"
        first_digest = hashlib.sha256(raw_string.encode('utf-8')).hexdigest()
        final_sign = hashlib.sha256((first_digest + self.api_secret).encode('utf-8')).hexdigest()
        return final_sign

    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float, risk_usd: float = 10.0) -> float:
        """
        Calcula el tamaño exacto de la posición para arriesgar exactamente $10 USD.
        Fórmula: Cantidad = Riesgo / Distancia de Stop Loss
        """
        risk_distance = abs(entry_price - stop_loss)
        if risk_distance <= 0:
            return 0.001
        
        qty = risk_usd / risk_distance
        
        # Redondear según el par
        if "BTC" in symbol:
            return round(qty, 3)
        elif "ETH" in symbol:
            return round(qty, 2)
        elif "SOL" in symbol or "AVAX" in symbol or "LINK" in symbol or "BNB" in symbol:
            return round(qty, 1)
        elif "DOGE" in symbol or "XRP" in symbol or "SUI" in symbol:
            return round(qty, 0)
        elif "PEPE" in symbol:
            return round(qty, -3)
        return round(qty, 2)

    def execute_order(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        risk_usd: float = 10.0
    ) -> Dict[str, Any]:
        """
        Ejecuta una orden de futuros en Bitunix con SL y TP vinculados.
        Si las claves API no están configuradas, opera en MODO SIMULACIÓN seguro.
        """
        clean_symbol = symbol.replace("/", "").replace("-", "").upper()
        qty = self.calculate_position_size(clean_symbol, entry_price, stop_loss, risk_usd)
        side = "BUY" if direction.upper() == "LONG" else "SELL"

        # MODO SIMULACIÓN (Si aún no se han configurado las claves privadas)
        if not self.is_configured():
            sim_id = f"SIM-{uuid.uuid4().hex[:8].upper()}"
            logger.info(f"🟢 [MODO SIMULACIÓN BITUNIX]: Orden {clean_symbol} {direction} creada.")
            logger.info(f"   Entrada: ${entry_price} | SL: ${stop_loss} | TP: ${take_profit} | Riesgo: ${risk_usd} USD (Cant: {qty})")
            return {
                "success": True,
                "mode": "SIMULATION",
                "order_id": sim_id,
                "symbol": clean_symbol,
                "direction": direction,
                "entry_price": entry_price,
                "qty": qty,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "msg": f"Orden SIMULADA con éxito en {clean_symbol} (Riesgo ${risk_usd} USD)"
            }

        # MODO REAL BITUNIX API
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
                logger.info(f"✔ Orden REAL enviada a Bitunix con éxito: {data}")
                return {
                    "success": True,
                    "mode": "REAL",
                    "order_id": data.get("data", {}).get("orderId", ""),
                    "symbol": clean_symbol,
                    "direction": direction,
                    "msg": f"Orden REAL enviada a Bitunix: {clean_symbol} {direction}"
                }
            else:
                logger.error(f"Bitunix rechazó la orden: {data}")
                return {"success": False, "mode": "REAL", "msg": data.get("msg", "Error desconocido en Bitunix")}
        except Exception as e:
            logger.error(f"Excepción al enviar orden a Bitunix: {e}")
            return {"success": False, "mode": "REAL", "msg": str(e)}

trader = BitunixTrader()

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

# Especificaciones exactas del mercado de futuros de Bitunix (basePrecision, minTradeVolume, quotePrecision)
SYMBOL_SPECS: Dict[str, Dict[str, Any]] = {
    "BTCUSDT": {"base_prec": 4, "min_vol": 0.0001, "quote_prec": 1},
    "ETHUSDT": {"base_prec": 3, "min_vol": 0.003, "quote_prec": 2},
    "SOLUSDT": {"base_prec": 2, "min_vol": 0.1, "quote_prec": 2},
    "XRPUSDT": {"base_prec": 1, "min_vol": 2.0, "quote_prec": 4},
    "DOGEUSDT": {"base_prec": 0, "min_vol": 53.0, "quote_prec": 5},
    "BNBUSDT": {"base_prec": 2, "min_vol": 0.01, "quote_prec": 2},
    "SUIUSDT": {"base_prec": 1, "min_vol": 10.0, "quote_prec": 4},
    "ADAUSDT": {"base_prec": 0, "min_vol": 15.0, "quote_prec": 4},
    "AVAXUSDT": {"base_prec": 0, "min_vol": 1.0, "quote_prec": 3},
    "LINKUSDT": {"base_prec": 2, "min_vol": 0.1, "quote_prec": 3},
    "NEARUSDT": {"base_prec": 0, "min_vol": 10.0, "quote_prec": 3},
    "TAOUSDT": {"base_prec": 3, "min_vol": 0.02, "quote_prec": 2},
    "1000PEPEUSDT": {"base_prec": 0, "min_vol": 790.0, "quote_prec": 7},
    "ONDOUSDT": {"base_prec": 1, "min_vol": 20.0, "quote_prec": 4},
    "ENAUSDT": {"base_prec": 0, "min_vol": 10.0, "quote_prec": 5},
    "WLDUSDT": {"base_prec": 0, "min_vol": 6.0, "quote_prec": 4},
    "UNIUSDT": {"base_prec": 0, "min_vol": 2.0, "quote_prec": 3},
    "XAUUSDT": {"base_prec": 3, "min_vol": 0.002, "quote_prec": 2},
    "XAGUSDT": {"base_prec": 3, "min_vol": 0.1, "quote_prec": 2},
    "CLUSDT": {"base_prec": 1, "min_vol": 0.1, "quote_prec": 2},
    "SPCXUSDT": {"base_prec": 2, "min_vol": 0.05, "quote_prec": 2},
    "NVDAUSDT": {"base_prec": 2, "min_vol": 0.01, "quote_prec": 2},
    "TSLAUSDT": {"base_prec": 2, "min_vol": 0.02, "quote_prec": 2},
}

def get_symbol_spec(symbol: str) -> Dict[str, Any]:
    clean = symbol.replace("/", "").replace("-", "").upper()
    if clean in SYMBOL_SPECS:
        return SYMBOL_SPECS[clean]
    for k, v in SYMBOL_SPECS.items():
        if clean in k or k in clean:
            return v
    return {"base_prec": 2, "min_vol": 0.01, "quote_prec": 2}

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
        account_equity: float = 50.0,
        default_risk_pct: float = 0.02
    ):
        self.api_key = api_key if api_key is not None else settings.BITUNIX_API_KEY
        self.api_secret = api_secret if api_secret is not None else settings.BITUNIX_API_SECRET
        self.base_url = settings.BITUNIX_REST_URL
        self.max_leverage_notional = getattr(settings, "MAX_LEVERAGE_NOTIONAL", 2.5)
        self.max_margin_pct = getattr(settings, "MAX_MARGIN_PCT_PER_TRADE", 0.08)
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
        
        # 1. Cantidad teórica según riesgo de pérdida en Stop Loss
        calculated_qty = target_risk / risk_distance
        
        # 2. Hard Leverage Cap (Tope Nocional Máximo, e.g. 2.5x notional)
        max_allowed_notional = equity * self.max_leverage_notional
        max_allowed_qty_notional = max_allowed_notional / entry_price
        
        # 3. Límite de Margen Retenido (Máx 8% del balance retenido a 10x)
        max_margin_usd = equity * self.max_margin_pct
        max_notional_margin = max_margin_usd * 10.0
        max_allowed_qty_margin = max_notional_margin / entry_price
        
        # Seleccionar la cantidad más segura que respete todos los filtros
        final_qty = min(calculated_qty, max_allowed_qty_notional, max_allowed_qty_margin)

        spec = get_symbol_spec(symbol)
        base_prec = spec.get("base_prec", 2)
        min_vol = spec.get("min_vol", 0.01)

        # Respetar el volumen mínimo del exchange
        final_qty = max(final_qty, min_vol)

        if base_prec == 0:
            return float(round(final_qty))
        return round(final_qty, base_prec)

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
        if clean_symbol == "PEPEUSDT":
            clean_symbol = "1000PEPEUSDT"

        spec = get_symbol_spec(clean_symbol)
        quote_prec = spec.get("quote_prec", 2)
        base_prec = spec.get("base_prec", 2)

        qty = self.calculate_position_size(clean_symbol, entry_price, stop_loss, risk_usd=actual_risk_usd, current_equity=equity)
        notional_value = qty * entry_price
        leverage_used = notional_value / equity
        estimated_margin = notional_value / 10.0 # Margen estimado retenido por Bitunix (10x colateral)
        side = "BUY" if direction.upper() == "LONG" else "SELL"

        if not self.is_configured():
            sim_id = f"SIM-{uuid.uuid4().hex[:8].upper()}"
            logger.info(f"🟢 [SIMULACIÓN BITUNIX]: {clean_symbol} {direction} | Cantidad: {qty} ({leverage_used:.1f}x) | Margen: ${estimated_margin:.2f}")
            return {
                "success": True,
                "mode": "SIMULATION",
                "order_id": sim_id,
                "symbol": clean_symbol,
                "direction": direction,
                "entry_price": entry_price,
                "qty": qty,
                "notional": round(notional_value, 2),
                "estimated_margin": round(estimated_margin, 2),
                "leverage": round(leverage_used, 1),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_usd": actual_risk_usd,
                "msg": f"Orden Virtual ejecutada: {qty} {clean_symbol} (Margen: ${estimated_margin:.2f} USDT, Riesgo SL: ${actual_risk_usd:.2f} USD)"
            }

        url = f"{self.base_url}/api/v1/futures/trade/place_order"
        nonce = uuid.uuid4().hex
        timestamp = str(int(time.time() * 1000))

        formatted_entry = f"{entry_price:.{quote_prec}f}"
        formatted_sl = f"{stop_loss:.{quote_prec}f}"
        formatted_tp = f"{take_profit:.{quote_prec}f}"
        formatted_qty = f"{int(qty)}" if base_prec == 0 else f"{qty:.{base_prec}f}"

        payload = {
            "symbol": clean_symbol,
            "side": side,
            "orderType": "LIMIT",
            "tradeSide": "OPEN",
            "price": formatted_entry,
            "qty": formatted_qty,
            "slPrice": formatted_sl,
            "tpPrice": formatted_tp
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
                logger.info(f"✔ Orden REAL enviada a Bitunix: ID {order_id} | Margen: ${estimated_margin:.2f} USDT")
                return {
                    "success": True,
                    "mode": "REAL",
                    "order_id": order_id,
                    "symbol": clean_symbol,
                    "direction": direction,
                    "qty": qty,
                    "notional": round(notional_value, 2),
                    "estimated_margin": round(estimated_margin, 2),
                    "risk_usd": actual_risk_usd,
                    "msg": f"Orden REAL enviada a Bitunix: {qty} {clean_symbol} (Margen: ${estimated_margin:.2f} USDT, Riesgo SL: ${actual_risk_usd:.2f} USD)"
                }
            else:
                logger.error(f"Bitunix rechazó la orden: {data}")
                return {
                    "success": False,
                    "mode": "REAL",
                    "order_id": "N/A",
                    "symbol": clean_symbol,
                    "direction": direction,
                    "qty": qty,
                    "notional": round(notional_value, 2),
                    "estimated_margin": round(estimated_margin, 2),
                    "risk_usd": actual_risk_usd,
                    "msg": data.get("msg", f"Error en Bitunix (code {data.get('code')})")
                }
        except Exception as e:
            logger.error(f"Excepción en Bitunix: {e}")
            return {
                "success": False,
                "mode": "REAL",
                "order_id": "N/A",
                "symbol": clean_symbol,
                "direction": direction,
                "qty": qty,
                "notional": round(notional_value, 2),
                "estimated_margin": round(estimated_margin, 2),
                "risk_usd": actual_risk_usd,
                "msg": str(e)
            }

    def get_open_positions(self) -> list:
        """Consulta posiciones abiertas directamente en Bitunix."""
        if not self.is_configured():
            return []
        url = f"{self.base_url}/api/v1/futures/position/get_pending_positions"
        nonce = uuid.uuid4().hex
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(nonce, timestamp, "")
        headers = {
            "api-key": self.api_key,
            "nonce": nonce,
            "timestamp": timestamp,
            "sign": signature,
            "Content-Type": "application/json"
        }
        try:
            res = requests.get(url, headers=headers, timeout=8)
            res.raise_for_status()
            data = res.json()
            if data.get("code") == 0:
                return data.get("data", [])
            return []
        except Exception as e:
            logger.error(f"Error consultando posiciones en Bitunix: {e}")
            return []

    def close_position(self, position_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Cierra una posición abierta en Bitunix inmediatamente al precio de mercado (Flash Close)."""
        if not self.is_configured():
            return {"success": True, "mode": "SIMULATION", "msg": f"Posición {position_id} cerrada en simulación"}

        url = f"{self.base_url}/api/v1/futures/trade/flash_close_position"
        nonce = uuid.uuid4().hex
        timestamp = str(int(time.time() * 1000))
        payload = {"positionId": str(position_id)}
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
            res = requests.post(url, data=body_str, headers=headers, timeout=8)
            res.raise_for_status()
            data = res.json()
            if data.get("code") == 0:
                logger.info(f"✔ Posición {position_id} ({symbol or ''}) CERRADA en Bitunix con éxito")
                return {"success": True, "mode": "REAL", "position_id": position_id, "msg": "Posición cerrada al mercado"}
            else:
                logger.error(f"Bitunix rechazó cerrar posición {position_id}: {data}")
                return {"success": False, "mode": "REAL", "msg": data.get("msg", "Error cerrando posición")}
        except Exception as e:
            logger.error(f"Excepción al cerrar posición {position_id} en Bitunix: {e}")
            return {"success": False, "mode": "REAL", "msg": str(e)}

    def close_all_positions(self) -> list:
        """Cierra todas las posiciones abiertas en Bitunix."""
        positions = self.get_open_positions()
        results = []
        for p in positions:
            pid = p.get("positionId")
            sym = p.get("symbol")
            if pid:
                r = self.close_position(pid, sym)
                results.append(r)
        return results

trader = BitunixTrader()


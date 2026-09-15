from typing import Dict, Any, Optional

class BitunixParser:
    """Parsea y normaliza los mensajes recibidos del WebSocket y REST de Bitunix."""

    @staticmethod
    def parse_kline_ws(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extrae la información de una vela OHLCV desde un mensaje WebSocket."""
        if data.get("ch") != "kline" and "kline" not in data.get("channel", ""):
            return None

        kdata = data.get("data", {})
        if not kdata:
            return None

        return {
            "symbol": data.get("symbol", kdata.get("s", "")),
            "timestamp": kdata.get("t", kdata.get("time")),
            "open": float(kdata.get("o", kdata.get("open", 0))),
            "high": float(kdata.get("h", kdata.get("high", 0))),
            "low": float(kdata.get("l", kdata.get("low", 0))),
            "close": float(kdata.get("c", kdata.get("close", 0))),
            "volume": float(kdata.get("v", kdata.get("vol", 0))),
            "is_closed": kdata.get("is_closed", False)
        }

import logging
import requests
import pandas as pd
from typing import Dict, Any, Optional
from src.exchanges.base import BaseExchangeClient
from config.settings import settings

logger = logging.getLogger("BitunixClient")

class BitunixClient(BaseExchangeClient):
    """Cliente REST para interactuar con la API de Futuros de Bitunix."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or settings.BITUNIX_REST_URL
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "TradiCopilot/1.0",
            "Accept": "application/json"
        })

    def get_historical_klines(self, symbol: str, interval: str = "15m", limit: int = 200) -> pd.DataFrame:
        """
        Descarga velas históricas (Klines) de futuros de Bitunix.
        Soporta paginación automática hacia atrás con endTime para límites superiores a 200.
        Retorna DataFrame con columnas: [timestamp, open, high, low, close, volume].
        """
        clean_symbol = symbol.replace("/", "").replace("-", "").upper()
        url = f"{self.base_url}/api/v1/futures/market/kline"
        all_raw_klines = []
        end_time = None
        remaining = limit

        try:
            while remaining > 0:
                batch_limit = min(200, remaining)
                params = {
                    "symbol": clean_symbol,
                    "interval": interval,
                    "limit": batch_limit
                }
                if end_time:
                    params["endTime"] = end_time

                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if "data" not in data or not data["data"]:
                    break

                batch = data["data"]
                if not isinstance(batch, list) or len(batch) == 0:
                    break

                all_raw_klines.extend(batch)
                remaining -= len(batch)

                # Extraer el timestamp más antiguo del lote para la siguiente página
                min_ts = min(int(x.get("time", x.get("t", 0))) for x in batch if (x.get("time") or x.get("t")))
                if min_ts <= 0 or (end_time is not None and min_ts >= end_time):
                    break
                end_time = min_ts - 1

                if len(batch) < batch_limit:
                    break

            if not all_raw_klines:
                logger.warning(f"No se recibieron datos de velas para {clean_symbol} en {interval}")
                return pd.DataFrame()

            df = pd.DataFrame(all_raw_klines)
            
            # Mapeo de nombres según la API de Bitunix
            column_mapping = {
                "time": "timestamp",
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "baseVol": "volume",
                "quoteVol": "quote_volume",
                "vol": "volume",
                "v": "volume"
            }
            df = df.rename(columns=column_mapping)

            if "volume" not in df.columns:
                df["volume"] = 0.0

            # Conversión numérica de columnas
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="ms")
            # Deduplicar y ordenar cronológicamente
            df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            return df[["timestamp", "open", "high", "low", "close", "volume"]].tail(limit).reset_index(drop=True)

        except Exception as e:
            logger.error(f"Error al obtener velas de Bitunix para {clean_symbol}: {e}")
            return pd.DataFrame()

    def get_order_book(self, symbol: str, limit: int = 15) -> Dict[str, Any]:
        """Obtiene la profundidad del libro de órdenes de Bitunix."""
        clean_symbol = symbol.replace("/", "").replace("-", "").upper()
        url = f"{self.base_url}/api/v1/futures/market/depth"
        params = {
            "symbol": clean_symbol,
            "limit": limit
        }

        try:
            response = self.session.get(url, params=params, timeout=8)
            response.raise_for_status()
            data = response.json()
            return data.get("data", {})
        except Exception as e:
            logger.error(f"Error al obtener order book de Bitunix para {clean_symbol}: {e}")
            return {}

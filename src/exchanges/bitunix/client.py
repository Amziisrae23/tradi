import logging
import time
import uuid
import hashlib
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
        self.api_key = settings.BITUNIX_API_KEY
        self.api_secret = settings.BITUNIX_API_SECRET
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "TradiCopilot/1.0",
            "Accept": "application/json"
        })

    def _sign(self, nonce: str, timestamp: str, body_str: str = "") -> str:
        """Genera firma SHA-256 doble para endpoints privados de Bitunix."""
        raw = f"{nonce}{timestamp}{self.api_key}{body_str}"
        first = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return hashlib.sha256((first + self.api_secret).encode("utf-8")).hexdigest()

    def get_account_balance(self) -> float:
        """
        Consulta el balance de USDT disponible en la cuenta de futuros de Bitunix.
        Retorna el equity total (USDT) o el valor por defecto de settings si falla.
        """
        if not self.api_key or not self.api_secret:
            logger.debug("Sin API Key — usando capital por defecto de settings.")
            return settings.FIXED_RISK_USD / 0.02  # Reverse del 2%

        try:
            url = f"{self.base_url}/api/v1/futures/account"
            nonce = uuid.uuid4().hex
            timestamp = str(int(time.time() * 1000))
            sign = self._sign(nonce, timestamp)

            headers = {
                "api-key": self.api_key,
                "nonce": nonce,
                "timestamp": timestamp,
                "sign": sign,
                "Content-Type": "application/json"
            }

            response = self.session.get(url, headers=headers, timeout=8)
            response.raise_for_status()
            data = response.json()

            if data.get("code") == 0:
                raw_acc = data.get("data", {})
                equity = None
                if isinstance(raw_acc, list):
                    for item in raw_acc:
                        if isinstance(item, dict):
                            eq = (
                                item.get("equity") or
                                item.get("totalEquity") or
                                item.get("availableBalance") or
                                item.get("walletBalance") or
                                item.get("available") or
                                item.get("balance")
                            )
                            if eq is not None and float(eq) > 0:
                                equity = eq
                                break
                    # Si no encontramos con balance positivo, tomar el primero si existe
                    if equity is None and len(raw_acc) > 0 and isinstance(raw_acc[0], dict):
                        equity = raw_acc[0].get("equity") or raw_acc[0].get("availableBalance") or raw_acc[0].get("balance")
                elif isinstance(raw_acc, dict):
                    equity = (
                        raw_acc.get("equity") or
                        raw_acc.get("totalEquity") or
                        raw_acc.get("availableBalance") or
                        raw_acc.get("walletBalance") or
                        raw_acc.get("balance")
                    )

                if equity is not None:
                    try:
                        equity_float = float(equity)
                        if equity_float > 0.5:
                            logger.info(f"💰 Balance real de Bitunix: ${equity_float:,.2f} USDT")
                            return equity_float
                        else:
                            fallback = getattr(settings, "INITIAL_CAPITAL", 50.0)
                            logger.info(f"ℹ️ Balance en Bitunix es $0.00 USDT — usando capital configurado de ${fallback:.2f} USDT")
                            return fallback
                    except (ValueError, TypeError):
                        pass

            logger.warning(f"No se detectó balance activo en Bitunix ({data.get('msg', 'sin fondos')})")

        except Exception as e:
            logger.warning(f"Error consultando balance de Bitunix: {e}")

        # Fallback: leer capital configurado (por defecto $50.0)
        fallback = getattr(settings, "INITIAL_CAPITAL", 50.0)
        logger.info(f"Usando capital configurado: ${fallback:.2f} USDT")
        return fallback

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

                response = self.session.get(url, params=params, timeout=8)
                response.raise_for_status()
                data = response.json()

                if "data" not in data or not data["data"]:
                    break

                batch = data["data"]
                if not isinstance(batch, list) or len(batch) == 0:
                    break

                all_raw_klines.extend(batch)
                remaining -= len(batch)

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

            column_mapping = {
                "time": "timestamp", "t": "timestamp",
                "o": "open", "h": "high", "l": "low", "c": "close",
                "baseVol": "volume", "quoteVol": "quote_volume",
                "vol": "volume", "v": "volume"
            }
            df = df.rename(columns=column_mapping)

            if "volume" not in df.columns:
                df["volume"] = 0.0

            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="ms")
            df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            return df[["timestamp", "open", "high", "low", "close", "volume"]].tail(limit).reset_index(drop=True)

        except Exception as e:
            logger.error(f"Error al obtener velas de Bitunix para {clean_symbol}: {e}")
            return pd.DataFrame()

    def get_order_book(self, symbol: str, limit: int = 15) -> Dict[str, Any]:
        """Obtiene la profundidad del libro de órdenes de Bitunix."""
        clean_symbol = symbol.replace("/", "").replace("-", "").upper()
        url = f"{self.base_url}/api/v1/futures/market/depth"
        params = {"symbol": clean_symbol, "limit": limit}

        try:
            response = self.session.get(url, params=params, timeout=8)
            response.raise_for_status()
            data = response.json()
            return data.get("data", {})
        except Exception as e:
            logger.error(f"Error al obtener order book de Bitunix para {clean_symbol}: {e}")
            return {}


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

                response = self.session.get(url, params=params, timeout=8)
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

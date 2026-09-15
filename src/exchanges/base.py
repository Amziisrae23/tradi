from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any, Optional

class BaseExchangeClient(ABC):
    """Interfaz base para conectores de exchanges."""

    @abstractmethod
    def get_historical_klines(self, symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
        """Descarga velas históricas en formato DataFrame."""
        pass

    @abstractmethod
    def get_order_book(self, symbol: str, limit: int = 15) -> Dict[str, Any]:
        """Obtiene la profundidad del libro de órdenes."""
        pass

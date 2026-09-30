import os
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseModel):
    # Bitunix API Config
    BITUNIX_API_KEY: str = os.getenv("BITUNIX_API_KEY", "")
    BITUNIX_API_SECRET: str = os.getenv("BITUNIX_API_SECRET", "")
    BITUNIX_REST_URL: str = os.getenv("BITUNIX_REST_URL", "https://fapi.bitunix.com")
    BITUNIX_WS_URL: str = os.getenv("BITUNIX_WS_URL", "wss://fapi.bitunix.com/public/")

    # Telegram Bot Config
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # Gemini AI Config
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Top 10 Monitored Liquid Cryptocurrency Futures Pairs
    DEFAULT_SYMBOLS: List[str] = [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "DOGEUSDT",
        "SUIUSDT",
        "ADAUSDT",
        "AVAXUSDT",
        "LINKUSDT",
        "BNBUSDT"
    ]
    
    # Timeframes for Multi-Timeframe Analysis
    HTF_INTERVAL: str = "4h"   # Higher Timeframe (Tendencia Macro)
    MTF_INTERVAL: str = "15m"  # Intermediate Timeframe (Setup / POI)
    LTF_INTERVAL: str = "5m"   # Lower Timeframe (Entrada / Gatillo)

    # Risk Management
    FIXED_RISK_USD: float = 10.0        # $10 USD de riesgo fijo por operación (2% de $500)
    MIN_RISK_REWARD_RATIO: float = 2.0  # R:R mínimo para emitir señal
    MIN_ML_CONFIDENCE: float = 75.0     # 75% de confianza mínima del modelo ML
    ATR_PERIOD: int = 14
    ATR_BUFFER_MULT: float = 0.35       # Buffer de seguridad para Stop Loss
    
    # Output paths
    CHARTS_DIR: str = os.path.join(os.getcwd(), "output", "charts")

settings = Settings()

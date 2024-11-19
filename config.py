from dotenv import load_dotenv
import os
from typing import Dict


class Config:
    def __init__(self):
        load_dotenv()
        self._load_credentials()
        self._load_trading_config()

    def _load_credentials(self):
        self.BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
        self.BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
        self.TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
        self.TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
        self.TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")
        self.ALERT_TO_NUMBER = os.getenv("ALERT_TO_NUMBER")
        self.CMC_API_KEY = os.getenv("CMC_API_KEY")

        if not self.BINANCE_API_KEY or not self.BINANCE_API_SECRET:
            raise ValueError("Binance credentials not found in environment variables")
        if not self.CMC_API_KEY:
            raise ValueError("CoinMarketCap API key not found in environment variables")

    def _load_trading_config(self):
        self.TRADING_CONFIG: Dict = {
            "market_data": {
                "top_coins": 10,        # Número de top coins a analizar
                "include_trending": True # Incluir monedas en tendencia
            },
            "intervals": {
                "klines": "4h",
                "analysis": "1h"
            },
            "limits": {
                "klines": 120,
                "volume": 500000,
                "trades": 2000
            },
            "filters": {
                "min_market_cap": 1000000,  # Capitalización mínima (1M USD)
                "min_volume_24h": 500000,   # Volumen mínimo 24h (500k USD)
                "exclude_stablecoins": True  # Excluir stablecoins del análisis
            }
        }

config = Config()

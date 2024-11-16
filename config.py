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

        if not self.BINANCE_API_KEY or not self.BINANCE_API_SECRET:
            raise ValueError("Binance credentials not found in environment variables")

    def _load_trading_config(self):
        self.TRADING_CONFIG: Dict = {
            "default_symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
            "intervals": {
                "klines": "4h",
                "analysis": "1h"
            },
            "limits": {
                "klines": 120,
                "volume": 500000,
                "trades": 2000
            }
        }

config = Config()

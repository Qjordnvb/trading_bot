# core/binance_client.py
import requests
import time
import hmac
import hashlib
import urllib.parse
from typing import Dict, List, Union
import numpy as np
from utils.console_colors import ConsoleColors

class BinanceClient:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.binance.com/api/v3"

    def _generate_signature(self, params: Dict) -> str:
        query_string = urllib.parse.urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _make_request(self, method: str, endpoint: str,
                     signed: bool = False, params: Dict = None) -> Union[List, Dict]:
        try:
            url = f"{self.base_url}{endpoint}"
            headers = {"X-MBX-APIKEY": self.api_key}
            params = params or {}

            if signed:
                params['timestamp'] = int(time.time() * 1000)
                params['signature'] = self._generate_signature(params)

            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(ConsoleColors.error(f"Error en request a {endpoint}: {str(e)}"))
            return [] if isinstance(e.response.json(), list) else {}

    def get_exchange_info(self) -> Dict:
        return self._make_request("GET", "/exchangeInfo")

    def get_ticker_24h(self, symbol: str = None) -> Union[List, Dict]:
        params = {"symbol": symbol} if symbol else {}
        return self._make_request("GET", "/ticker/24hr", params=params)

    def get_klines(self, symbol: str, interval: str = "1h", limit: int = 100) -> List[Dict]:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        data = self._make_request("GET", "/klines", params=params)

        formatted_data = []
        for candle in data:
            formatted_data.append({
                "timestamp": candle[0],
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5]),
                "close_time": candle[6],
                "quote_volume": float(candle[7]),
                "trades": int(candle[8]),
                "taker_buy_base_volume": float(candle[9]),
                "taker_buy_quote_volume": float(candle[10])
            })
        return formatted_data

    def get_ticker_price(self, symbol: str = None) -> Union[List, Dict]:
        params = {"symbol": symbol} if symbol else {}
        return self._make_request("GET", "/ticker/price", params=params)

    def calculate_market_metrics(self, symbol: str) -> Dict:
        try:
            ticker_24h = self._make_request("GET", "/ticker/24hr", params={"symbol": symbol})
            klines = self.get_klines(symbol, interval="1d", limit=7)

            if not ticker_24h or not klines:
                return {}

            current_price = float(ticker_24h["lastPrice"])
            volume_24h = float(ticker_24h["quoteVolume"])
            change_24h = float(ticker_24h["priceChangePercent"])

            closes = [candle["close"] for candle in klines]
            returns = [
                (closes[i] - closes[i - 1]) / closes[i - 1]
                for i in range(1, len(closes))
            ]
            volatility = (sum(r * r for r in returns) / len(returns)) ** 0.5 * 100

            return {
                "current_price": current_price,
                "volume_24h": volume_24h,
                "change_24h": change_24h,
                "volatility_7d": volatility,
                "high_24h": float(ticker_24h["highPrice"]),
                "low_24h": float(ticker_24h["lowPrice"])
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando métricas para {symbol}: {str(e)}"))
            return {}

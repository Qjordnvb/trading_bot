import requests
import time
import hmac
import hashlib
from typing import Dict, List, Union
from enum import Enum
from datetime import datetime
import os
from dotenv import load_dotenv
from colorama import init, Fore, Back, Style

# Inicializar colorama
init(autoreset=True)

# Cargar variables de entorno
load_dotenv()
API_KEY = os.getenv("GATE_IO_API_KEY")
API_SECRET = os.getenv("GATE_IO_API_SECRET")

if not API_KEY or not API_SECRET:
    print(
        f"{Fore.RED}Error: No se encontraron las credenciales de Gate.io en las variables de entorno{Style.RESET_ALL}"
    )
    raise ValueError("Credenciales no encontradas")


class ConsoleColors:
    """Clase para manejar los colores y estilos de la consola"""

    @staticmethod
    def header(text: str) -> str:
        return f"{Fore.CYAN}{Style.BRIGHT}{text}{Style.RESET_ALL}"

    @staticmethod
    def success(text: str) -> str:
        return f"{Fore.GREEN}{text}{Style.RESET_ALL}"

    @staticmethod
    def error(text: str) -> str:
        return f"{Fore.RED}{text}{Style.RESET_ALL}"

    @staticmethod
    def warning(text: str) -> str:
        return f"{Fore.YELLOW}{text}{Style.RESET_ALL}"

    @staticmethod
    def info(text: str) -> str:
        return f"{Fore.BLUE}{text}{Style.RESET_ALL}"

    @staticmethod
    def highlight(text: str) -> str:
        return f"{Fore.MAGENTA}{text}{Style.RESET_ALL}"

    @staticmethod
    def price_change(change: float) -> str:
        return (
            f"{Fore.GREEN if change >= 0 else Fore.RED}{change:+.2f}%{Style.RESET_ALL}"
        )


class TradingSignal(Enum):
    BUY = "COMPRAR"
    SELL = "VENDER"
    HOLD = "MANTENER"


class GateIOClient:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.gateio.ws/api/v4"

    def _generate_signature(
        self, method: str, url: str, query_string: str = "", body: str = ""
    ) -> Dict[str, str]:
        """Genera la firma para autenticación"""
        t = str(int(time.time()))
        m = hashlib.sha512()
        m.update((query_string + body).encode("utf-8"))
        hashed_payload = m.hexdigest()
        s = "%s\n%s\n%s\n%s\n%s" % (method, url, query_string, hashed_payload, t)
        sign = hmac.new(
            self.api_secret.encode("utf-8"), s.encode("utf-8"), hashlib.sha512
        ).hexdigest()

        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "KEY": self.api_key,
            "Timestamp": t,
            "SIGN": sign,
        }

    def _make_request(
        self, method: str, endpoint: str, params: Dict = None
    ) -> Union[List, Dict]:
        """Método centralizado para hacer requests"""
        try:
            url = f"{self.base_url}{endpoint}"
            query_string = "&".join(f"{k}={v}" for k, v in (params or {}).items())
            headers = self._generate_signature(method, endpoint, query_string)

            response = requests.request(
                method=method, url=url, headers=headers, params=params
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error en request a {endpoint}: {str(e)}")
            return [] if isinstance(e.response.json(), list) else {}

    def get_spot_tickers(self) -> List[Dict]:
        """Obtiene información de todos los pares de trading en spot"""
        return self._make_request("GET", "/spot/tickers")

    def get_spot_currency_pairs(self) -> List[Dict]:
        """Obtiene información de todos los pares de trading disponibles"""
        return self._make_request("GET", "/spot/currency_pairs")

    def get_candlesticks(
        self, currency_pair: str, interval: str = "1h", limit: int = 100
    ) -> List[Dict]:
        """
        Obtiene datos históricos de precios y los devuelve en formato estructurado
        intervals: ['10s', '1m', '5m', '15m', '30m', '1h', '4h', '8h', '1d', '7d']
        """
        params = {"currency_pair": currency_pair, "interval": interval, "limit": limit}
        data = self._make_request("GET", "/spot/candlesticks", params)

        # Convertir los datos en un formato más manejable
        formatted_data = []
        for candle in data:
            if len(candle) >= 8:
                formatted_data.append(
                    {
                        "timestamp": int(candle[0]),
                        "volume": float(candle[1]),
                        "close": float(candle[2]),
                        "high": float(candle[3]),
                        "low": float(candle[4]),
                        "open": float(candle[5]),
                        "amount": float(candle[6]),
                        "is_complete": candle[7] == "true",
                    }
                )
        return formatted_data

    def get_ticker(self, currency_pair: str) -> Dict:
        """Obtiene información detallada de un par específico"""
        tickers = self._make_request(
            "GET", "/spot/tickers", {"currency_pair": currency_pair}
        )
        return tickers[0] if tickers else {}

    def get_trade_history(self, currency_pair: str, limit: int = 100) -> List[Dict]:
        """Obtiene el historial de trades recientes"""
        params = {"currency_pair": currency_pair, "limit": limit}
        return self._make_request("GET", "/spot/trades", params)

    def calculate_market_metrics(self, currency_pair: str) -> Dict:
        """Calcula métricas de mercado usando datos disponibles"""
        try:
            # Obtener datos necesarios
            ticker = self.get_ticker(currency_pair)
            candlesticks = self.get_candlesticks(currency_pair, interval="1d", limit=7)

            if not ticker or not candlesticks:
                return {}

            # Calcular métricas
            current_price = float(ticker["last"])
            volume_24h = float(ticker["quote_volume"])
            change_24h = float(ticker["change_percentage"])

            # Calcular volatilidad usando los últimos 7 días
            prices = [candle["close"] for candle in candlesticks]
            volatility = self._calculate_volatility(prices)

            return {
                "current_price": current_price,
                "volume_24h": volume_24h,
                "change_24h": change_24h,
                "volatility_7d": volatility,
                "high_24h": float(ticker["high_24h"]),
                "low_24h": float(ticker["low_24h"]),
            }
        except Exception as e:
            print(f"Error calculando métricas para {currency_pair}: {str(e)}")
            return {}

    @staticmethod
    def _calculate_volatility(prices: List[float]) -> float:
        """Calcula la volatilidad de una lista de precios"""
        if len(prices) < 2:
            return 0

        returns = [
            (prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))
        ]
        return (
            sum(r * r for r in returns) / len(returns)
        ) ** 0.5 * 100  # Volatilidad en porcentaje


def format_number(number: float, decimals: int = 2) -> str:
    """Formatea números con separadores de miles y decimales específicos"""
    return f"{number:,.{decimals}f}"


def print_dict(data: Dict, indent: int = 0, title: str = None):
    """Imprime un diccionario de forma formateada y colorida"""
    if title:
        print("\n" + " " * indent + ConsoleColors.header(f"=== {title} ==="))

    for key, value in data.items():
        key_str = ConsoleColors.info(f"{key}: ")
        if isinstance(value, (int, float)):
            if "price" in key.lower():
                value_str = ConsoleColors.highlight(f"${format_number(value, 8)}")
            elif "volume" in key.lower():
                value_str = ConsoleColors.success(f"${format_number(value, 2)}")
            elif "change" in key.lower() or "percent" in key.lower():
                value_str = ConsoleColors.price_change(value)
            else:
                value_str = format_number(value)
        else:
            value_str = str(value)

        print(" " * indent + f"{key_str}{value_str}")


def test_client():
    """Función para probar el cliente con salida colorida"""
    try:
        client = GateIOClient(API_KEY, API_SECRET)

        # Header principal
        print(ConsoleColors.header("\n=== PRUEBA DE CONEXIÓN CON GATE.IO ==="))

        # Probar obtención de tickers
        print(ConsoleColors.header("\n▶ Probando obtención de tickers..."))
        tickers = client.get_spot_tickers()
        print(ConsoleColors.success(f"✓ Tickers obtenidos: {len(tickers)}"))
        if tickers:
            print(ConsoleColors.info("\nEjemplo de ticker:"))
            print_dict(tickers[0], indent=2)

        # Probar obtención de pares de trading
        print(ConsoleColors.header("\n▶ Probando obtención de pares de trading..."))
        pairs = client.get_spot_currency_pairs()
        print(ConsoleColors.success(f"✓ Pares obtenidos: {len(pairs)}"))
        if pairs:
            print(ConsoleColors.info("\nEjemplo de par:"))
            print_dict(pairs[0], indent=2)

        # Probar obtención de candlesticks
        print(
            ConsoleColors.header(
                "\n▶ Probando obtención de candlesticks para BTC_USDT..."
            )
        )
        candlesticks = client.get_candlesticks("BTC_USDT")
        print(ConsoleColors.success(f"✓ Candlesticks obtenidos: {len(candlesticks)}"))
        if candlesticks:
            print(ConsoleColors.info("\nEjemplo de candlestick:"))
            print_dict(candlesticks[0], indent=2)

        # Probar cálculo de métricas de mercado
        print(
            ConsoleColors.header(
                "\n▶ Probando cálculo de métricas de mercado para BTC_USDT..."
            )
        )
        metrics = client.calculate_market_metrics("BTC_USDT")
        if metrics:
            print(ConsoleColors.info("\nMétricas de mercado:"))
            print_dict(metrics, indent=2)

        # Probar obtención de historial de trades
        print(
            ConsoleColors.header(
                "\n▶ Probando obtención de historial de trades para BTC_USDT..."
            )
        )
        trades = client.get_trade_history("BTC_USDT", limit=5)
        print(ConsoleColors.success(f"✓ Trades obtenidos: {len(trades)}"))
        if trades:
            print(ConsoleColors.info("\nEjemplo de trade:"))
            print_dict(trades[0], indent=2)

    except Exception as e:
        print(ConsoleColors.error(f"\n❌ Error en las pruebas: {str(e)}"))


if __name__ == "__main__":
    print(ConsoleColors.header("\n=== INICIANDO PRUEBAS DE GATE.IO API ==="))
    print(
        ConsoleColors.info("Fecha y hora: ")
        + ConsoleColors.highlight(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )

    test_client()

    print(ConsoleColors.header("\n=== PRUEBAS COMPLETADAS ===\n"))

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
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
import numpy as np


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

class EntryTiming(Enum):
    IMMEDIATE = "ENTRADA INMEDIATA"
    WAIT_DIP = "ESPERAR RETROCESO"
    WAIT_BREAKOUT = "ESPERAR RUPTURA"
    WAIT_CONSOLIDATION = "ESPERAR CONSOLIDACIÓN"
    NOT_RECOMMENDED = "ENTRADA NO RECOMENDADA"


class TradingSignal(Enum):
    BUY = "COMPRAR"
    SELL = "VENDER"
    HOLD = "MANTENER"


class MarketTrend(Enum):
    STRONG_UPTREND = "FUERTE TENDENCIA ALCISTA"
    UPTREND = "TENDENCIA ALCISTA"
    NEUTRAL = "NEUTRAL"
    DOWNTREND = "TENDENCIA BAJISTA"
    STRONG_DOWNTREND = "FUERTE TENDENCIA BAJISTA"

class SignalStrength(Enum):
    STRONG = "FUERTE"
    MODERATE = "MODERADA"
    WEAK = "DÉBIL"

class TradeRecommendation:
    def __init__(self, signal: TradingSignal, strength: SignalStrength, reasons: List[str],
                 entry_price: float = None, stop_loss: float = None, take_profit: float = None):
        self.signal = signal
        self.strength = strength
        self.reasons = reasons
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit

class TimingWindow:
    def __init__(self, timing: EntryTiming, timeframe: str, target_price: float = None,
                 confidence: float = 0.0, conditions: List[str] = None):
        self.timing = timing
        self.timeframe = timeframe
        self.target_price = target_price
        self.confidence = confidence
        self.conditions = conditions or []

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



class MarketAnalyzer:
    def __init__(self, client):
        self.client = client
        self.thresholds = {
            "momentum": {
                "strong_buy": 15,    # 15% incremento
                "buy": 5,            # 5% incremento
                "strong_sell": -15,  # 15% decremento
                "sell": -5          # 5% decremento
            },
            "volume": {
                "significant": 2,    # 2x promedio de volumen
                "moderate": 1.5,     # 1.5x promedio de volumen
                "low": 0.5          # 0.5x promedio de volumen
            },
            "rsi": {
                "strong_oversold": 20,
                "oversold": 30,
                "overbought": 70,
                "strong_overbought": 80
            }
        }
        self.timing_thresholds = {
            # RSI thresholds
            "oversold_rsi": 30,
            "overbought_rsi": 70,

            # Precio thresholds
            "price_support": 0.05,    # 5% desde soporte
            "price_resistance": 0.05,  # 5% desde resistencia

            # Volumen thresholds
            "volume_spike": 2.0,       # 2x del promedio
            "volume_significant": 1.5,  # 1.5x del promedio

            # Volatilidad thresholds
            "consolidation_range": 0.02,  # 2% de rango para consolidación
            "high_volatility": 0.05,      # 5% indica alta volatilidad
            "low_volatility": 0.01,       # 1% indica baja volatilidad

            # Momentum thresholds
            "strong_momentum": 0.15,    # 15% de movimiento
            "weak_momentum": 0.05,      # 5% de movimiento

            # Tendencia thresholds
            "trend_strength": {
                "strong": 0.15,        # 15% de movimiento
                "moderate": 0.08,      # 8% de movimiento
                "weak": 0.03           # 3% de movimiento
            },

            # Volumen thresholds detallados
            "volume_levels": {
                "high": 2.0,           # 2x promedio
                "moderate": 1.5,       # 1.5x promedio
                "low": 0.5             # 0.5x promedio
            }
        }


    def _calculate_volatility(self, candlesticks: List[Dict], period: int = 14) -> float:
        """
        Calcula la volatilidad del mercado usando True Range y desviación estándar
        """
        try:
            if not candlesticks or len(candlesticks) < period:
                return 0.0

            # Extraer precios
            highs = [float(candle['high']) for candle in candlesticks]
            lows = [float(candle['low']) for candle in candlesticks]
            closes = [float(candle['close']) for candle in candlesticks]

            # Calcular True Ranges
            true_ranges = []
            for i in range(1, len(candlesticks)):
                true_range = max(
                    highs[i] - lows[i],  # Alto - Bajo actual
                    abs(highs[i] - closes[i-1]),  # Alto actual - Cierre previo
                    abs(lows[i] - closes[i-1])  # Bajo actual - Cierre previo
                )
                true_ranges.append(true_range)

            # Calcular ATR (Average True Range)
            atr = sum(true_ranges[-period:]) / period if true_ranges else 0

            # Calcular cambios porcentuales diarios
            daily_returns = [
                (closes[i] - closes[i-1]) / closes[i-1]
                for i in range(1, len(closes))
            ]

            # Calcular desviación estándar de los retornos
            if daily_returns:
                mean_return = sum(daily_returns) / len(daily_returns)
                squared_diff = sum((r - mean_return) ** 2 for r in daily_returns)
                std_dev = (squared_diff / len(daily_returns)) ** 0.5
            else:
                std_dev = 0

            # Combinar ATR y desviación estándar para obtener la volatilidad
            volatility = (atr / closes[-1] + std_dev) / 2

            return volatility

        except Exception as e:
            print(ConsoleColors.warning(f"Error calculando volatilidad: {str(e)}"))
            return 0.0

    def _format_volatility(self, volatility: float) -> str:
        """Formatea la volatilidad como porcentaje y determina su nivel"""
        volatility_percentage = volatility * 100

        if volatility_percentage > 5:
            return f"ALTA ({volatility_percentage:.1f}%)"
        elif volatility_percentage > 2:
            return f"MEDIA ({volatility_percentage:.1f}%)"
        else:
            return f"BAJA ({volatility_percentage:.1f}%)"

    def analyze_entry_timing(self, candlesticks: List[Dict], current_price: float) -> TimingWindow:
        """Analiza el mejor momento para entrar en una posición"""
        try:
            if not candlesticks or len(candlesticks) < 50:
                return TimingWindow(
                    EntryTiming.NOT_RECOMMENDED,
                    "N/A",
                    conditions=["Datos insuficientes para análisis"]
                )

            # Analizar diferentes aspectos del timing
            rsi = self._calculate_rsi(candlesticks)
            volume_analysis = self._analyze_volume(candlesticks)
            support_resistance = self._calculate_support_resistance(candlesticks)
            volatility = self._calculate_volatility(candlesticks)
            pattern = self._identify_pattern(candlesticks)

            # Agregar el nivel de volatilidad a las condiciones
            volatility_level = self._format_volatility(volatility)

            # Determinar el mejor momento de entrada
            timing, timeframe, target, confidence, conditions = self._determine_timing(
                current_price, rsi, volume_analysis, support_resistance,
                volatility, pattern
            )

            # Agregar la volatilidad a las condiciones
            conditions.append(f"Volatilidad: {volatility_level}")

            return TimingWindow(
                timing=timing,
                timeframe=timeframe,
                target_price=target,
                confidence=confidence,
                conditions=conditions
            )

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando timing: {str(e)}"))
            return TimingWindow(
                EntryTiming.NOT_RECOMMENDED,
                "N/A",
                conditions=["Error en análisis de timing"]
            )

    def _evaluate_entry_type(self, **kwargs) -> Tuple[EntryTiming, float]:
        """Evalúa el tipo de entrada más apropiado"""
        trend_score = kwargs.get('trend_score', 0)
        rsi = kwargs.get('rsi', 50)
        volume_analysis = kwargs.get('volume_analysis', {})
        pattern = kwargs.get('pattern', {})

        if rsi <= self.timing_thresholds['oversold_rsi'] and trend_score > 0:
            return EntryTiming.IMMEDIATE, 0.8
        elif rsi >= self.timing_thresholds['overbought_rsi']:
            return EntryTiming.WAIT_DIP, 0.7
        elif pattern.get('type') == 'bullish' and volume_analysis.get('is_increasing'):
            return EntryTiming.WAIT_BREAKOUT, 0.6
        elif abs(trend_score) < self.timing_thresholds['trend_strength']['weak']:
            return EntryTiming.WAIT_CONSOLIDATION, 0.5
        else:
            return EntryTiming.NOT_RECOMMENDED, 0.3

    def _calculate_optimal_timeframe(self, **kwargs) -> str:
        """Calcula el timeframe óptimo para la entrada"""
        volatility = kwargs.get('volatility', 0)
        volume_analysis = kwargs.get('volume_analysis', {})
        trend_score = kwargs.get('trend_score', 0)

        base_timeframe = "4-8 horas"

        # Ajustar por volatilidad
        if volatility > self.timing_thresholds['trend_strength']['strong']:
            base_timeframe = "12-24 horas"
        elif volatility < self.timing_thresholds['trend_strength']['weak']:
            base_timeframe = "2-4 horas"

        # Ajustar por volumen
        if volume_analysis.get('ratio', 1) > self.timing_thresholds['volume_levels']['high']:
            base_timeframe = self._adjust_timeframe(base_timeframe, 0.75)
        elif volume_analysis.get('ratio', 1) < self.timing_thresholds['volume_levels']['low']:
            base_timeframe = self._adjust_timeframe(base_timeframe, 1.5)

        return base_timeframe

    def _analyze_specific_conditions(self, **kwargs) -> List[str]:
        """Analiza condiciones específicas para la moneda"""
        conditions = []
        symbol = kwargs.get('symbol', '')
        trend_score = kwargs.get('trend_score', 0)
        momentum = kwargs.get('momentum', {})
        volume_analysis = kwargs.get('volume_analysis', {})
        timeframes = kwargs.get('timeframes', {})

        # Análisis de tendencia
        if trend_score > self.timing_thresholds['trend_strength']['strong']:
            conditions.append(f"Fuerte tendencia alcista ({trend_score:.1f}%)")
        elif trend_score < -self.timing_thresholds['trend_strength']['strong']:
            conditions.append(f"Fuerte tendencia bajista ({trend_score:.1f}%)")

        # Análisis de momentum
        if momentum.get('value', 0) > self.timing_thresholds['trend_strength']['moderate']:
            conditions.append(f"Momentum positivo fuerte ({momentum.get('value', 0):.1f})")
        elif momentum.get('value', 0) < -self.timing_thresholds['trend_strength']['moderate']:
            conditions.append(f"Momentum negativo fuerte ({momentum.get('value', 0):.1f})")

        # Análisis de volumen
        vol_ratio = volume_analysis.get('ratio', 1)
        if vol_ratio > self.timing_thresholds['volume_levels']['high']:
            conditions.append(f"Volumen excepcionalmente alto ({vol_ratio:.1f}x promedio)")
        elif vol_ratio < self.timing_thresholds['volume_levels']['low']:
            conditions.append(f"Volumen bajo ({vol_ratio:.1f}x promedio)")

        return conditions

    def _adjust_final_confidence(self, **kwargs) -> float:
        """Ajusta la confianza final basada en todos los factores"""
        base_confidence = kwargs.get('base_confidence', 0)
        conditions = kwargs.get('conditions', [])
        trend_score = kwargs.get('trend_score', 0)
        volatility = kwargs.get('volatility', 0)

        # Ajustes de confianza basados en condiciones
        confidence_adjustments = {
            "Fuerte tendencia": 0.2,
            "Momentum fuerte": 0.15,
            "Volumen alto": 0.1,
            "Patrón confirmado": 0.15
        }

        final_confidence = base_confidence
        for condition in conditions:
            for key, adjustment in confidence_adjustments.items():
                if key in condition:
                    final_confidence += adjustment

        # Ajuste por volatilidad
        if volatility > self.timing_thresholds['trend_strength']['strong']:
            final_confidence *= 0.8  # Reducir confianza en alta volatilidad

        return min(max(final_confidence, 0.0), 1.0)

    def _determine_timing(self, current_price: float, rsi: float,
                         volume_analysis: Dict, support_resistance: Dict,
                         volatility: float, pattern: Dict) -> Tuple[EntryTiming, str, float, float, List[str]]:
        """Determina el mejor momento para entrar basado en múltiples factores"""
        conditions = []
        confidence = 0.0
        timing = EntryTiming.NOT_RECOMMENDED
        timeframe = "12-24 horas"  # timeframe por defecto
        target_price = current_price

        try:
            # Ajustar timeframes basados en volatilidad
            high_volatility = volatility > self.timing_thresholds["consolidation_range"]
            timeframe_multiplier = 2 if high_volatility else 1

            # Análisis de RSI
            if rsi <= self.timing_thresholds["oversold_rsi"]:
                timing = EntryTiming.IMMEDIATE
                timeframe = "0-4 horas"
                confidence += 0.3
                conditions.append(f"RSI en sobreventa ({rsi:.2f})")
            elif rsi >= self.timing_thresholds["overbought_rsi"]:
                timing = EntryTiming.WAIT_DIP
                timeframe = "12-24 horas"
                confidence += 0.2
                conditions.append(f"RSI en sobrecompra ({rsi:.2f})")
            else:
                timing = EntryTiming.WAIT_CONSOLIDATION
                timeframe = "4-12 horas"
                confidence += 0.1

            # Análisis de volumen
            if volume_analysis.get('is_significant', False):
                confidence += 0.2
                if volume_analysis.get('is_increasing', False):
                    conditions.append("Volumen creciente")
                    if timing != EntryTiming.WAIT_DIP:
                        timing = EntryTiming.IMMEDIATE
                        timeframe = "0-4 horas"
                else:
                    conditions.append("Alto volumen pero decreciente")

            # Análisis de soporte/resistencia
            support = support_resistance.get('support', 0)
            resistance = support_resistance.get('resistance', 0)

            if current_price > 0 and support > 0:
                distance_to_support = (current_price - support) / support
                if distance_to_support <= self.timing_thresholds["price_support"]:
                    timing = EntryTiming.IMMEDIATE
                    timeframe = "0-4 horas"
                    target_price = support
                    confidence += 0.25
                    conditions.append(f"Precio cerca del soporte (${support:,.8f})")

            if current_price > 0 and resistance > 0:
                distance_to_resistance = (resistance - current_price) / current_price
                if distance_to_resistance <= self.timing_thresholds["price_resistance"]:
                    timing = EntryTiming.WAIT_BREAKOUT
                    timeframe = "6-12 horas"
                    target_price = resistance * 1.02
                    confidence += 0.15
                    conditions.append(f"Precio cerca de resistencia (${resistance:,.8f})")

            # Análisis de volatilidad
            if high_volatility:
                timeframe = self._adjust_timeframe(timeframe, timeframe_multiplier)
                conditions.append(f"Alta volatilidad ({volatility:.1%})")
                confidence -= 0.1
            else:
                conditions.append(f"Baja volatilidad ({volatility:.1%})")
                confidence += 0.1

            # Análisis de patrones
            pattern_type = pattern.get('type', 'neutral')
            if pattern_type == 'bullish':
                if timing != EntryTiming.IMMEDIATE:
                    timing = EntryTiming.WAIT_BREAKOUT
                    timeframe = "4-8 horas"
                confidence += 0.2
                conditions.append(f"Patrón alcista: {pattern.get('name', 'desconocido')}")
            elif pattern_type == 'bearish':
                timing = EntryTiming.WAIT_DIP
                timeframe = "12-24 horas"
                confidence -= 0.1
                conditions.append(f"Patrón bajista: {pattern.get('name', 'desconocido')}")

            # Ajustar confianza final
            confidence = min(max(confidence, 0.0), 1.0)

        except Exception as e:
            print(ConsoleColors.error(f"Error en determine_timing: {str(e)}"))
            return EntryTiming.NOT_RECOMMENDED, "N/A", current_price, 0.0, ["Error en análisis"]

        return timing, timeframe, target_price, confidence, conditions

    def _determine_precise_timing(self, current_price: float, symbol: str, **kwargs) -> Tuple[EntryTiming, str, float, float, List[str]]:
        """Determina el timing preciso basado en múltiples factores"""
        conditions = []
        confidence = 0.0

        # Extraer datos del análisis
        rsi = kwargs.get('rsi', 50)
        volume_analysis = kwargs.get('volume_analysis', {})
        volatility = kwargs.get('volatility', 0)
        trend_analysis = kwargs.get('trend_analysis', {})
        support_resistance = kwargs.get('support_resistance', {})
        pattern = kwargs.get('pattern', {})
        momentum = kwargs.get('momentum', {})
        timeframes = kwargs.get('timeframes', {})

        # Análisis inicial de tendencia
        trend_score = self._evaluate_trend_context(timeframes)

        # Determinar tipo de entrada basado en el contexto
        timing, initial_confidence = self._evaluate_entry_type(
            trend_score=trend_score,
            rsi=rsi,
            volume_analysis=volume_analysis,
            pattern=pattern
        )
        confidence += initial_confidence

        # Ajustar timeframe basado en volatilidad y volumen
        timeframe = self._calculate_optimal_timeframe(
            volatility=volatility,
            volume_analysis=volume_analysis,
            trend_score=trend_score
        )

        # Calcular precio objetivo
        target_price = self._calculate_target_price(
            current_price=current_price,
            trend_score=trend_score,
            support_resistance=support_resistance,
            pattern=pattern
        )

        # Analizar condiciones específicas
        specific_conditions = self._analyze_specific_conditions(
            symbol=symbol,
            trend_score=trend_score,
            momentum=momentum,
            volume_analysis=volume_analysis,
            timeframes=timeframes
        )
        conditions.extend(specific_conditions)

        # Ajustar confianza final
        confidence = self._adjust_final_confidence(
            base_confidence=confidence,
            conditions=conditions,
            trend_score=trend_score,
            volatility=volatility
        )

        return timing, timeframe, target_price, confidence, conditions

    def _evaluate_trend_context(self, timeframes: Dict) -> float:
        """Evalúa el contexto general de la tendencia"""
        short_term = timeframes.get('short', {}).get('trend', {}).get('strength', 0)
        medium_term = timeframes.get('medium', {}).get('trend', {}).get('strength', 0)
        long_term = timeframes.get('long', {}).get('trend', {}).get('strength', 0)

        # Ponderación de diferentes timeframes
        trend_score = (
            short_term * 0.5 +
            medium_term * 0.3 +
            long_term * 0.2
        )

        return trend_score

    def _analyze_timeframe(self, candles: List[Dict]) -> Dict:
        """Analiza un timeframe específico para tendencias y patrones"""
        closes = [float(candle['close']) for candle in candles]
        volumes = [float(candle['volume']) for candle in candles]

        return {
            'trend': self._calculate_trend_direction(closes),
            'momentum': self._calculate_momentum_strength(closes),
            'volume_trend': self._calculate_volume_trend(volumes),
            'volatility': self._calculate_volatility(candles),
            'key_levels': self._identify_key_levels(candles)
        }

    def _calculate_trend_direction(self, prices: List[float]) -> Dict:
        """Calcula la dirección y fuerza de la tendencia"""
        if len(prices) < 2:
            return {'direction': 'neutral', 'strength': 0}

        # Calcular cambios porcentuales
        changes = [(prices[i] - prices[i-1])/prices[i-1] * 100
                  for i in range(1, len(prices))]

        # Calcular tendencia
        trend_strength = sum(changes) / len(changes)

        if trend_strength > self.timing_thresholds['trend_strength']['strong']:
            direction = 'strong_bullish'
        elif trend_strength > self.timing_thresholds['trend_strength']['moderate']:
            direction = 'moderate_bullish'
        elif trend_strength < -self.timing_thresholds['trend_strength']['strong']:
            direction = 'strong_bearish'
        elif trend_strength < -self.timing_thresholds['trend_strength']['moderate']:
            direction = 'moderate_bearish'
        else:
            direction = 'neutral'

        return {
            'direction': direction,
            'strength': abs(trend_strength)
        }

    def _adjust_timeframe(self, timeframe: str, multiplier: int) -> str:
        """Ajusta el timeframe según el multiplicador"""
        try:
            # Extraer números del timeframe
            import re
            numbers = re.findall(r'\d+', timeframe)
            if len(numbers) == 2:
                min_time, max_time = map(int, numbers)
                return f"{min_time * multiplier}-{max_time * multiplier} horas"
            return timeframe
        except:
            return timeframe

    def _calculate_support_resistance(self, candlesticks: List[Dict]) -> Dict:
        """Calcula niveles de soporte y resistencia"""
        try:
            closes = [float(candle['close']) for candle in candlesticks]
            highs = [float(candle['high']) for candle in candlesticks]
            lows = [float(candle['low']) for candle in candlesticks]

            # Método simple para identificar niveles
            support = min(lows[-20:])  # Soporte basado en mínimos recientes
            resistance = max(highs[-20:])  # Resistencia basada en máximos recientes

            return {
                'support': support,
                'resistance': resistance
            }
        except Exception:
            return {
                'support': min(closes) if closes else 0,
                'resistance': max(closes) if closes else 0
            }

    def _identify_pattern(self, candlesticks: List[Dict]) -> Dict:
        """Identifica patrones de precio"""
        try:
            closes = [float(candle['close']) for candle in candlesticks]

            # Identificar tendencia reciente
            short_term_trend = (closes[-1] - closes[-3]) / closes[-3]

            if len(closes) >= 5:
                # Identificar patrones simples
                if all(closes[i] <= closes[i+1] for i in range(-5, -1)):
                    return {'type': 'bullish', 'name': 'Tendencia Alcista Continua'}
                elif all(closes[i] >= closes[i+1] for i in range(-5, -1)):
                    return {'type': 'bearish', 'name': 'Tendencia Bajista Continua'}
                elif closes[-1] > closes[-2] > closes[-3] and closes[-4] > closes[-3]:
                    return {'type': 'bullish', 'name': 'Rebote en V'}
                elif closes[-1] < closes[-2] < closes[-3] and closes[-4] < closes[-3]:
                    return {'type': 'bearish', 'name': 'Caída en V Invertida'}

            return {'type': 'neutral', 'name': 'Sin patrón claro'}

        except Exception:
            return {'type': 'neutral', 'name': 'Error en identificación'}



    def _analyze_trend(self, candlesticks: List[Dict]) -> MarketTrend:
        """Analiza la tendencia del mercado"""
        closes = [float(candle['close']) for candle in candlesticks]

        # Calcular EMAs
        ema20 = self._calculate_ema(closes, 20)
        ema50 = self._calculate_ema(closes, 50)

        # Determinar tendencia
        if ema20[-1] > ema50[-1] and closes[-1] > ema20[-1]:
            return MarketTrend.STRONG_UPTREND
        elif ema20[-1] > ema50[-1]:
            return MarketTrend.UPTREND
        elif ema20[-1] < ema50[-1] and closes[-1] < ema20[-1]:
            return MarketTrend.STRONG_DOWNTREND
        elif ema20[-1] < ema50[-1]:
            return MarketTrend.DOWNTREND
        return MarketTrend.NEUTRAL



    def _calculate_trade_levels(self, current_price: float, trend: MarketTrend,
                              candlesticks: List[Dict]) -> Dict:
        """Calcula niveles de entrada, stop loss y take profit"""
        try:
            # Calcular ATR para volatilidad
            atr = self._calculate_atr(candlesticks)

            # Obtener niveles de soporte/resistencia
            support_resistance = self._calculate_support_resistance(candlesticks)

            # Calcular porcentajes base según la tendencia
            if trend in [MarketTrend.STRONG_UPTREND, MarketTrend.UPTREND]:
                stop_loss_pct = 0.15  # 15% para el stop loss
                take_profit_pct = 0.25  # 25% para take profit
            else:
                stop_loss_pct = 0.10  # 10% para el stop loss
                take_profit_pct = 0.20  # 20% para take profit

            # Ajustar porcentajes según volatilidad (ATR)
            volatility_factor = (atr / current_price)
            stop_loss_pct = min(stop_loss_pct * (1 + volatility_factor), 0.20)  # Máximo 20%
            take_profit_pct = max(take_profit_pct * (1 + volatility_factor), 0.15)  # Mínimo 15%

            # Calcular niveles base
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit = current_price * (1 + take_profit_pct)

            # Ajustar con niveles de soporte/resistencia
            nearest_support = support_resistance.get('support', stop_loss)
            nearest_resistance = support_resistance.get('resistance', take_profit)

            # Refinar stop loss y take profit
            if nearest_support > stop_loss:
                stop_loss = nearest_support * 0.98  # 2% debajo del soporte

            if nearest_resistance < take_profit:
                take_profit = nearest_resistance * 1.02  # 2% arriba de la resistencia

            # Asegurar ratio riesgo/beneficio mínimo de 1:1.5
            risk = current_price - stop_loss
            reward = take_profit - current_price
            if reward / risk < 1.5:
                take_profit = current_price + (risk * 1.5)

            return {
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': (take_profit - current_price) / (current_price - stop_loss)
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando niveles de trading: {str(e)}"))
            # Niveles por defecto más conservadores
            return {
                'entry_price': current_price,
                'stop_loss': current_price * 0.85,  # -15%
                'take_profit': current_price * 1.25,  # +25%
                'risk_reward_ratio': 1.67
            }

    def _generate_recommendation(self, trend: MarketTrend, volume: Dict,
                               momentum: Dict, rsi: float, price: float) -> Tuple[TradingSignal, SignalStrength, List[str]]:
        """Genera una recomendación de trading basada en el análisis"""
        reasons = []
        score = 0

        # Analizar tendencia
        if trend == MarketTrend.STRONG_UPTREND:
            score += 2
            reasons.append(f"Fuerte tendencia alcista")
        elif trend == MarketTrend.STRONG_DOWNTREND:
            score -= 2
            reasons.append(f"Fuerte tendencia bajista")

        # Analizar volumen
        if volume['is_significant'] and volume['is_increasing']:
            score += 1
            reasons.append(f"Alto volumen con tendencia creciente ({volume['ratio']:.1f}x promedio)")

        # Analizar momentum
        if momentum['is_strong'] and momentum['is_positive']:
            score += 1
            reasons.append(f"Fuerte momentum positivo (24h: {momentum['short_term']:.1f}%, 7d: {momentum['medium_term']:.1f}%)")
        elif momentum['is_strong'] and not momentum['is_positive']:
            score -= 1
            reasons.append(f"Fuerte momentum negativo (24h: {momentum['short_term']:.1f}%, 7d: {momentum['medium_term']:.1f}%)")

        # Analizar RSI
        if rsi < self.thresholds['rsi']['strong_oversold']:
            score += 2
            reasons.append(f"RSI indica sobreventa fuerte ({rsi:.1f})")
        elif rsi < self.thresholds['rsi']['oversold']:
            score += 1
            reasons.append(f"RSI indica sobreventa ({rsi:.1f})")
        elif rsi > self.thresholds['rsi']['strong_overbought']:
            score -= 2
            reasons.append(f"RSI indica sobrecompra fuerte ({rsi:.1f})")
        elif rsi > self.thresholds['rsi']['overbought']:
            score -= 1
            reasons.append(f"RSI indica sobrecompra ({rsi:.1f})")

        # Determinar señal y fuerza
        if score >= 3:
            signal = TradingSignal.BUY
            strength = SignalStrength.STRONG
        elif score >= 1:
            signal = TradingSignal.BUY
            strength = SignalStrength.MODERATE
        elif score <= -3:
            signal = TradingSignal.SELL
            strength = SignalStrength.STRONG
        elif score <= -1:
            signal = TradingSignal.SELL
            strength = SignalStrength.MODERATE
        else:
            signal = TradingSignal.HOLD
            strength = SignalStrength.WEAK
            reasons.append("No hay señales claras de trading")

        return signal, strength, reasons

    @staticmethod
    def _calculate_ema(values: List[float], period: int) -> List[float]:
        """Calcula el EMA (Exponential Moving Average)"""
        multiplier = 2 / (period + 1)
        ema = [values[0]]

        for price in values[1:]:
            ema.append((price * multiplier) + (ema[-1] * (1 - multiplier)))

        return ema

    @staticmethod
    def _calculate_rsi(candlesticks: List[Dict], period: int = 14) -> float:
        """Calcula el RSI (Relative Strength Index)"""
        closes = [float(candle['close']) for candle in candlesticks]
        deltas = np.diff(closes)

        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gain[-period:])
        avg_loss = np.mean(loss[-period:])

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _calculate_atr(candlesticks: List[Dict], period: int = 14) -> float:
        """Calcula el ATR (Average True Range)"""
        highs = [float(candle['high']) for candle in candlesticks]
        lows = [float(candle['low']) for candle in candlesticks]
        closes = [float(candle['close']) for candle in candlesticks]

        tr = []
        for i in range(1, len(candlesticks)):
            tr.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            ))

        return sum(tr[-period:]) / period

    def analyze_trading_opportunity(self, symbol: str) -> Optional[TradeRecommendation]:
        """Analiza una oportunidad de trading específica"""
        try:
            # Obtener datos históricos
            candlesticks = self.client.get_candlesticks(symbol, interval='4h', limit=120)
            ticker = self.client.get_ticker(symbol)

            # Validación de datos suficientes
            if not candlesticks or len(candlesticks) < 50:  # Requerimos al menos 50 velas
                print(ConsoleColors.warning(
                    f"Datos históricos insuficientes para {symbol}"
                ))
                return None

            if not ticker:
                print(ConsoleColors.warning(
                    f"No se pudo obtener información actual de {symbol}"
                ))
                return None

            # Analizar diferentes aspectos
            trend = self._analyze_trend(candlesticks)
            volume_analysis = self._analyze_volume(candlesticks)
            momentum = self._analyze_momentum(candlesticks)

            try:
                rsi = self._calculate_rsi(candlesticks)
            except Exception:
                rsi = 50  # Valor neutral si no se puede calcular

            # Obtener precio actual
            current_price = float(ticker['last'])

            # Generar recomendación
            signal, strength, reasons = self._generate_recommendation(
                trend, volume_analysis, momentum, rsi, current_price
            )

            # Calcular niveles de entrada, stop loss y take profit
            try:
                levels = self._calculate_trade_levels(current_price, trend, candlesticks)
            except Exception:
                levels = {
                    'stop_loss': current_price * 0.95,  # 5% por defecto
                    'take_profit': current_price * 1.15  # 15% por defecto
                }

            return TradeRecommendation(
                signal=signal,
                strength=strength,
                reasons=reasons,
                entry_price=current_price,
                stop_loss=levels['stop_loss'],
                take_profit=levels['take_profit']
            )

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando {symbol}: {str(e)}"))
            return None

    def _analyze_trend(self, candlesticks: List[Dict]) -> MarketTrend:
        """Analiza la tendencia del mercado"""
        try:
            closes = [float(candle['close']) for candle in candlesticks]

            if len(closes) < 50:  # Validación adicional
                return MarketTrend.NEUTRAL

            # Calcular EMAs
            ema20 = self._calculate_ema(closes[-20:], 20)
            ema50 = self._calculate_ema(closes[-50:], 50)

            # Determinar tendencia
            if len(ema20) > 0 and len(ema50) > 0:
                if ema20[-1] > ema50[-1] and closes[-1] > ema20[-1]:
                    return MarketTrend.STRONG_UPTREND
                elif ema20[-1] > ema50[-1]:
                    return MarketTrend.UPTREND
                elif ema20[-1] < ema50[-1] and closes[-1] < ema20[-1]:
                    return MarketTrend.STRONG_DOWNTREND
                elif ema20[-1] < ema50[-1]:
                    return MarketTrend.DOWNTREND
            return MarketTrend.NEUTRAL
        except Exception:
            return MarketTrend.NEUTRAL

    def _analyze_volume(self, candlesticks: List[Dict]) -> Dict:
        """Analiza el patrón de volumen"""
        try:
            volumes = [float(candle['volume']) for candle in candlesticks]

            if len(volumes) < 20:  # Validación adicional
                return {
                    'ratio': 1.0,
                    'is_significant': False,
                    'is_increasing': False,
                    'average': sum(volumes) / len(volumes) if volumes else 0
                }

            avg_volume = sum(volumes[-20:]) / 20
            current_volume = volumes[-1]

            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

            return {
                'ratio': volume_ratio,
                'is_significant': volume_ratio > self.thresholds['volume']['significant'],
                'is_increasing': sum(volumes[-3:]) > sum(volumes[-6:-3]),
                'average': avg_volume
            }
        except Exception as e:
            print(ConsoleColors.warning(f"Error en análisis de volumen: {str(e)}"))
            return {
                'ratio': 1.0,
                'is_significant': False,
                'is_increasing': False,
                'average': 0
            }

    def _analyze_momentum(self, candlesticks: List[Dict]) -> Dict:
        """Analiza el momentum del precio"""
        try:
            closes = [float(candle['close']) for candle in candlesticks]

            if len(closes) < 42:  # Validación para 7 días de datos
                return {
                    'short_term': 0,
                    'medium_term': 0,
                    'is_strong': False,
                    'is_positive': False
                }

            # Calcular cambios porcentuales
            change_24h = ((closes[-1] - closes[-6]) / closes[-6]) * 100  # 6 periodos de 4h = 24h
            change_7d = ((closes[-1] - closes[-42]) / closes[-42]) * 100  # 42 periodos de 4h = 7d

            return {
                'short_term': change_24h,
                'medium_term': change_7d,
                'is_strong': abs(change_24h) > self.thresholds['momentum']['strong_buy'],
                'is_positive': change_24h > 0 and change_7d > 0
            }
        except Exception:
            return {
                'short_term': 0,
                'medium_term': 0,
                'is_strong': False,
                'is_positive': False
            }

def format_timing_window(timing: TimingWindow) -> None:
    """Formatea y muestra la ventana de tiempo para entrada"""
    print(ConsoleColors.header("\nAnálisis de Timing"))

    # Mostrar recomendación de timing
    print(ConsoleColors.info("Recomendación: ") +
          ConsoleColors.highlight(timing.timing.value))

    # Información específica según el tipo de timing
    if timing.timing == EntryTiming.WAIT_CONSOLIDATION:
        print(ConsoleColors.info("\nDetalles de la Consolidación:"))
        print(ConsoleColors.success("  • Esperar hasta que el precio establezca un rango claro"))
        print(ConsoleColors.success("  • Observar el volumen para confirmación"))
        print(ConsoleColors.success("  • Buscar formación de patrones técnicos"))

        print(ConsoleColors.info("\nSeñales de Entrada:"))
        print(ConsoleColors.warning("  • Ruptura del rango de consolidación"))
        print(ConsoleColors.warning("  • Aumento significativo en volumen"))
        print(ConsoleColors.warning("  • Confirmación de dirección"))

    # Mostrar timeframe
    print(ConsoleColors.info("\nVentana de Tiempo: ") +
          ConsoleColors.highlight(timing.timeframe))

    # Mostrar nivel de confianza
    confidence_color = (ConsoleColors.success if timing.confidence > 0.7
                       else ConsoleColors.warning if timing.confidence > 0.4
                       else ConsoleColors.error)
    print(ConsoleColors.info("Nivel de Confianza: ") +
          confidence_color(f"{timing.confidence:.1f}%"))

    # Mostrar condiciones actuales
    if timing.conditions:
        print(ConsoleColors.info("\nCondiciones Actuales:"))
        for condition in timing.conditions:
            print(ConsoleColors.success(f"  • {condition}"))

    if timing.target_price:
        print(ConsoleColors.info("\nNiveles de Precio:"))
        print(ConsoleColors.highlight(f"  • Precio Objetivo: ${timing.target_price:,.8f}"))

def format_trade_recommendation(recommendation: TradeRecommendation) -> None:
    """Formatea y muestra una recomendación de trading"""
    if not recommendation:
        print(ConsoleColors.error("No se pudo generar recomendación"))
        return

    print(ConsoleColors.header("\nRecomendación de Trading"))
    print(ConsoleColors.info("Señal: ") +
          ConsoleColors.highlight(f"{recommendation.signal.value} ({recommendation.strength.value})"))

    print(ConsoleColors.info("\nNiveles de Trading:"))
    print(f"  Precio de Entrada: {ConsoleColors.highlight(f'${recommendation.entry_price:,.2f}')}")
    print(f"  Stop Loss: {ConsoleColors.error(f'${recommendation.stop_loss:,.2f}')} " +
          f"({((recommendation.stop_loss - recommendation.entry_price) / recommendation.entry_price * 100):,.2f}%)")
    print(f"  Take Profit: {ConsoleColors.success(f'${recommendation.take_profit:,.2f}')} " +
          f"({((recommendation.take_profit - recommendation.entry_price) / recommendation.entry_price * 100):,.2f}%)")

    print(ConsoleColors.info("\nRazones:"))
    for reason in recommendation.reasons:
        print(ConsoleColors.success(f"  • {reason}"))

def test_market_analyzer():
    """Función para probar el MarketAnalyzer"""
    try:
        client = GateIOClient(API_KEY, API_SECRET)
        analyzer = MarketAnalyzer(client)

        # Lista de símbolos a analizar
        symbols = ["BTC_USDT", "ETH_USDT", "DOGE_USDT"]

        for symbol in symbols:
            print(ConsoleColors.header(f"\n=== ANÁLISIS DE {symbol} ==="))
            recommendation = analyzer.analyze_trading_opportunity(symbol)
            format_trade_recommendation(recommendation)

    except Exception as e:
        print(ConsoleColors.error(f"\n❌ Error en las pruebas: {str(e)}"))

if __name__ == "__main__":
    print(ConsoleColors.header("\n=== INICIANDO ANÁLISIS DE TRADING ==="))
    print(ConsoleColors.info("Fecha y hora: ") +
          ConsoleColors.highlight(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    test_market_analyzer()

    print(ConsoleColors.header("\n=== ANÁLISIS COMPLETADO ===\n"))


class MemeCoinAnalyzer:
    def __init__(self, client, market_analyzer):
        self.client = client
        self.market_analyzer = market_analyzer

        # Keywords para identificar meme coins
        self.meme_keywords = {
            "primary": ["doge", "shib", "pepe", "floki", "meme", "inu"],
            "secondary": [
                "elon", "wojak", "chad", "cat", "kitty", "safe",
                "moon", "rocket", "baby", "pup", "pug", "shiba",
                "akita", "corgi", "moon"
            ],
            "exclude": ["chain", "swap", "protocol", "finance"]
        }

        # Criterios de filtrado
        self.filters = {
            "min_volume_24h": 100000,  # Mínimo $100k de volumen en 24h
            "max_price": 1.0,          # Máximo $1 por unidad (típico de meme coins)
            "min_trades": 1000,        # Mínimo número de trades en 24h
        }

    def get_top_meme_coins(self, limit: int = 10) -> List[Dict]:
        """Identifica y analiza las mejores meme coins"""
        try:
            print(ConsoleColors.info("\nBuscando meme coins..."))

            # Obtener todos los pares de trading
            all_pairs = self.client.get_spot_currency_pairs()
            tickers = self.client.get_spot_tickers()

            # Crear un diccionario de tickers para búsqueda rápida
            ticker_dict = {ticker['currency_pair']: ticker for ticker in tickers}

            # Identificar meme coins
            meme_coins = []
            for pair in all_pairs:
                currency_pair = pair['id']
                base_currency = pair['base'].lower()

                # Verificar si es una meme coin
                if self._is_meme_coin(base_currency):
                    ticker = ticker_dict.get(currency_pair)
                    if ticker and self._meets_criteria(ticker):
                        # Analizar la oportunidad de trading
                        analysis = self._analyze_meme_coin(currency_pair, ticker)
                        if analysis:
                            meme_coins.append(analysis)

            # Ordenar por puntuación y obtener los mejores
            meme_coins.sort(key=lambda x: x['score'], reverse=True)
            top_memes = meme_coins[:limit]

            print(ConsoleColors.success(
                f"\nSe encontraron {len(top_memes)} meme coins prometedoras"
            ))

            return top_memes

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando meme coins: {str(e)}"))
            return []

    def _is_meme_coin(self, currency: str) -> bool:
        """Determina si una moneda es meme coin"""
        # Verificar palabras de exclusión
        if any(word in currency for word in self.meme_keywords["exclude"]):
            return False

        # Verificar keywords primarios (mayor peso)
        primary_match = any(
            keyword in currency for keyword in self.meme_keywords["primary"]
        )
        if primary_match:
            return True

        # Verificar keywords secundarios (requiere múltiples coincidencias)
        secondary_matches = sum(
            1 for keyword in self.meme_keywords["secondary"]
            if keyword in currency
        )
        return secondary_matches >= 2

    def _meets_criteria(self, ticker: Dict) -> bool:
        """Verifica si la meme coin cumple con los criterios básicos"""
        try:
            quote_volume = float(ticker.get('quote_volume', 0))
            price = float(ticker.get('last', 0))

            return (
                quote_volume >= self.filters["min_volume_24h"] and
                price <= self.filters["max_price"]
            )
        except (ValueError, TypeError):
            return False

    def _analyze_meme_coin(self, currency_pair: str, ticker: Dict) -> Optional[Dict]:
        """Analiza una meme coin específica"""
        try:
            # Verificar volumen mínimo
            if float(ticker.get('quote_volume', 0)) < self.filters["min_volume_24h"]:
                return None

            # Obtener recomendación de trading
            recommendation = self.market_analyzer.analyze_trading_opportunity(currency_pair)
            if not recommendation:
                return None

            # Obtener datos de candlesticks para análisis de timing
            candlesticks = self.client.get_candlesticks(currency_pair)
            if not candlesticks:
                print(ConsoleColors.warning(
                    f"No hay suficientes datos históricos para {currency_pair}"
                ))
                return None

            current_price = float(ticker['last'])

            # Analizar timing
            timing_analysis = self.market_analyzer.analyze_entry_timing(
                candlesticks, current_price
            )

            # Calcular puntuación
            score = self._calculate_meme_score(ticker, recommendation)

            return {
                'currency_pair': currency_pair,
                'price': current_price,
                'volume_24h': float(ticker['quote_volume']),
                'change_24h': float(ticker.get('change_percentage', 0)),
                'recommendation': recommendation,
                'timing_analysis': timing_analysis,  # Cambiado de 'timing' a 'timing_analysis'
                'score': score
            }

        except Exception as e:
            print(ConsoleColors.error(
                f"Error analizando {currency_pair}: {str(e)}"
            ))
            return None

    def _calculate_meme_score(self, ticker: Dict, recommendation: TradeRecommendation) -> float:
        """Calcula una puntuación para la meme coin"""
        score = 0

        # Factor de señal de trading
        signal_weight = {
            TradingSignal.BUY: 1,
            TradingSignal.HOLD: 0,
            TradingSignal.SELL: -1
        }
        score += signal_weight[recommendation.signal] * 2

        # Factor de fuerza de la señal
        strength_weight = {
            SignalStrength.STRONG: 2,
            SignalStrength.MODERATE: 1,
            SignalStrength.WEAK: 0
        }
        score += strength_weight[recommendation.strength]

        # Factor de volumen
        try:
            volume = float(ticker['quote_volume'])
            if volume > self.filters["min_volume_24h"] * 10:
                score += 2
            elif volume > self.filters["min_volume_24h"] * 5:
                score += 1
        except (ValueError, KeyError):
            pass

        # Factor de cambio de precio
        try:
            change = float(ticker['change_percentage'])
            if change > 20:
                score += 2
            elif change > 10:
                score += 1
            elif change < -20:
                score -= 2
            elif change < -10:
                score -= 1
        except (ValueError, KeyError):
            pass

        return score


def print_meme_coin_analysis(meme_coins: List[Dict]) -> None:
    print(ConsoleColors.header("\n=== TOP 10 MEME COINS ===\n"))

    for i, coin in enumerate(meme_coins, 1):
        try:
            print(ConsoleColors.header(f"\n{i}. {coin['currency_pair']}"))

            # Información básica
            print(ConsoleColors.info("Precio Actual: ") +
                  ConsoleColors.highlight(f"${coin['price']:.8f}"))
            print(ConsoleColors.info("Volumen 24h: ") +
                  ConsoleColors.success(f"${coin['volume_24h']:,.2f}"))
            print(ConsoleColors.info("Cambio 24h: ") +
                  ConsoleColors.price_change(coin['change_24h']))

            if coin.get('recommendation'):
                print("\nRecomendación de Trading")
                print(ConsoleColors.highlight(f"Señal: {coin['recommendation'].signal.value} ({coin['recommendation'].strength.value})\n"))

                # Niveles de Trading actualizados
                print(ConsoleColors.info("Niveles de Trading:"))
                print(ConsoleColors.info("  Precio de Entrada: ") +
                     ConsoleColors.highlight(f"${coin['recommendation'].entry_price:.8f}"))

                stop_loss_pct = ((coin['recommendation'].stop_loss - coin['recommendation'].entry_price)
                                / coin['recommendation'].entry_price * 100)
                print(ConsoleColors.info("  Stop Loss: ") +
                     ConsoleColors.error(f"${coin['recommendation'].stop_loss:.8f} ({stop_loss_pct:.2f}%)"))

                take_profit_pct = ((coin['recommendation'].take_profit - coin['recommendation'].entry_price)
                                 / coin['recommendation'].entry_price * 100)
                print(ConsoleColors.info("  Take Profit: ") +
                     ConsoleColors.success(f"${coin['recommendation'].take_profit:.8f} ({take_profit_pct:.2f}%)"))

                if coin['recommendation'].reasons:
                    print("\nRazones:")
                    for reason in coin['recommendation'].reasons:
                        print(ConsoleColors.success(f"  • {reason}"))

            if coin.get('timing_analysis'):
                print("\nAnálisis de Timing")
                print(ConsoleColors.highlight(f"Recomendación: {coin['timing_analysis'].timing.value}"))
                print(ConsoleColors.highlight(f"Timeframe: {coin['timing_analysis'].timeframe}"))
                print(ConsoleColors.highlight(f"Confianza: {coin['timing_analysis'].confidence:.1%}"))

                if coin['timing_analysis'].conditions:
                    print("\nCondiciones Actuales:")
                    for condition in coin['timing_analysis'].conditions:
                        print(ConsoleColors.success(f"  • {condition}"))

            print(ConsoleColors.info("\nPuntuación General: ") +
                  ConsoleColors.highlight(f"{coin['score']:.2f}"))
            print("-" * 50)

        except Exception as e:
            print(ConsoleColors.error(f"Error mostrando análisis para {coin.get('currency_pair', 'Unknown')}: {str(e)}"))
            continue

def test_meme_coin_analyzer():
    """Función para probar el análisis de meme coins"""
    try:
        client = GateIOClient(API_KEY, API_SECRET)
        market_analyzer = MarketAnalyzer(client)
        meme_analyzer = MemeCoinAnalyzer(client, market_analyzer)

        print(ConsoleColors.header("\n=== INICIANDO ANÁLISIS DE MEME COINS ==="))
        print(ConsoleColors.info("Buscando y analizando meme coins..."))

        top_memes = meme_analyzer.get_top_meme_coins()

        if not top_memes:
            print(ConsoleColors.warning("\nNo se encontraron meme coins que cumplan los criterios"))
            return

        print(ConsoleColors.success(f"\nSe encontraron {len(top_memes)} meme coins prometedoras"))
        print_meme_coin_analysis(top_memes)

    except Exception as e:
        print(ConsoleColors.error(f"\n❌ Error en análisis de meme coins: {str(e)}"))
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    print(ConsoleColors.header("\n=== INICIANDO ANÁLISIS DE MERCADO CRYPTO ==="))
    print(ConsoleColors.info("Fecha y hora: ") +
          ConsoleColors.highlight(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    test_meme_coin_analyzer()

    print(ConsoleColors.header("\n=== ANÁLISIS COMPLETADO ===\n"))

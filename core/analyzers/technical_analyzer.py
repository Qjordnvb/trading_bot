from typing import Dict, List, Optional
import numpy as np
from utils.console_colors import ConsoleColors

class TechnicalAnalyzer:
    def __init__(self):
        self.rsi_params = {
            "period": 14,
            "oversold": 30,
            "overbought": 70,
            "extreme_oversold": 20,
            "extreme_overbought": 80
        }

        self.macd_params = {
            "fast": 12,
            "slow": 26,
            "signal": 9,
            "min_hist_strength": 0.2
        }

        self.bollinger_params = {
            "period": 20,
            "std_dev": 2.0,
            "squeeze_threshold": 0.5,
            "expansion_threshold": 2.5
        }

    def analyze(self, candlesticks: List[Dict]) -> Dict:
        """
        Analiza los indicadores técnicos usando los datos en formato de lista/tupla
        """
        try:
            if not candlesticks or len(candlesticks) < 50:
                return {
                    'rsi': 50,
                    'macd': {'trend': 'neutral', 'histogram': 0, 'crossover': 'none'},
                    'bollinger': {'upper': 0, 'middle': 0, 'lower': 0},
                    'is_valid': False
                }

            closes = self._extract_closes(candlesticks)

            if len(closes) < 50:
                return self._get_default_analysis()

            rsi = self.calculate_rsi(closes[-14:])
            macd = self.calculate_macd(closes)
            bb = self.calculate_bollinger_bands(closes[-20:])

            return {
                'rsi': rsi,
                'macd': macd,
                'bollinger': bb,
                'is_valid': True,
                'last_price': closes[-1]
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error en análisis técnico: {str(e)}"))
            return self._get_default_analysis()

    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calcula el RSI (Relative Strength Index)"""
        try:
            prices = np.array(prices, dtype=float)

            if len(prices) < period + 1:
                return 50.0

            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])

            if avg_loss == 0:
                return 100.0
            if avg_gain == 0:
                return 0.0

            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

            return min(100.0, max(0.0, float(rsi)))

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando RSI: {str(e)}"))
            return 50.0

    def calculate_macd(self, prices: List[float]) -> Dict:
        """Calcula el MACD (Moving Average Convergence Divergence)"""
        try:
            # Calcular EMAs
            fast_ema = self._calculate_ema(prices, self.macd_params["fast"])
            slow_ema = self._calculate_ema(prices, self.macd_params["slow"])

            # Calcular línea MACD
            macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]

            # Calcular línea de señal
            signal_line = self._calculate_ema(macd_line, self.macd_params["signal"])

            # Calcular histograma
            histogram = [m - s for m, s in zip(macd_line, signal_line)]

            return {
                'macd': macd_line[-1],
                'signal': signal_line[-1],
                'histogram': histogram[-1],
                'trending_up': macd_line[-1] > signal_line[-1],
                'momentum_strength': abs(histogram[-1]),
                'crossover': self._determine_macd_crossover(macd_line, signal_line)
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando MACD: {str(e)}"))
            return {
                'macd': 0, 'signal': 0, 'histogram': 0,
                'trending_up': False, 'momentum_strength': 0,
                'crossover': 'none'
            }

    def calculate_bollinger_bands(self, prices: List[float]) -> Dict:
        """Calcula las bandas de Bollinger"""
        try:
            if len(prices) < self.bollinger_params["period"]:
                return {'upper': prices[-1], 'middle': prices[-1], 'lower': prices[-1]}

            sma = sum(prices[-self.bollinger_params["period"]:]) / self.bollinger_params["period"]
            std = np.std(prices[-self.bollinger_params["period"]:])

            upper = sma + (self.bollinger_params["std_dev"] * std)
            lower = sma - (self.bollinger_params["std_dev"] * std)

            bandwidth = (upper - lower) / sma

            return {
                'upper': upper,
                'middle': sma,
                'lower': lower,
                'bandwidth': bandwidth,
                'squeeze': bandwidth < self.bollinger_params["squeeze_threshold"],
                'expansion': bandwidth > self.bollinger_params["expansion_threshold"]
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando Bollinger Bands: {str(e)}"))
            return {'upper': prices[-1], 'middle': prices[-1], 'lower': prices[-1]}

    def _calculate_ema(self, values: List[float], period: int) -> List[float]:
        """Calcula EMA (Exponential Moving Average)"""
        if not values:
            return []

        ema = [values[0]]
        multiplier = 2 / (period + 1)

        for price in values[1:]:
            ema.append((price * multiplier) + (ema[-1] * (1 - multiplier)))

        return ema

    def _extract_closes(self, candlesticks: List[Dict]) -> List[float]:
        """Extrae los precios de cierre de las velas"""
        closes = []
        for candle in candlesticks:
            try:
                if isinstance(candle, dict):
                    closes.append(float(candle['close']))
                elif isinstance(candle, (list, tuple)):
                    closes.append(float(candle[4]))
            except (IndexError, KeyError, ValueError) as e:
                print(f"Error extrayendo precio de cierre: {e}")
                continue
        return closes

    def _determine_macd_crossover(self, macd_line: List[float], signal_line: List[float]) -> str:
        """Determina el tipo de cruce del MACD"""
        if len(macd_line) < 2 or len(signal_line) < 2:
            return 'none'

        if macd_line[-1] > signal_line[-1] and macd_line[-2] <= signal_line[-2]:
            return 'bullish'
        elif macd_line[-1] < signal_line[-1] and macd_line[-2] >= signal_line[-2]:
            return 'bearish'
        return 'none'

    def _get_default_analysis(self) -> Dict:
        """Retorna un análisis por defecto cuando hay error o datos insuficientes"""
        return {
            'rsi': 50,
            'macd': {'trend': 'neutral', 'histogram': 0, 'crossover': 'none'},
            'bollinger': {'upper': 0, 'middle': 0, 'lower': 0},
            'is_valid': False
        }

from typing import Dict, List, Optional
import numpy as np
from utils.console_colors import ConsoleColors

class VolatilityAnalyzer:
    def __init__(self):
        self.volatility_params = {
            "atr": {
                "period": 14,              # Período para ATR
                "high_threshold": 0.05,    # 5% ATR alto
                "medium_threshold": 0.03,  # 3% ATR medio
                "low_threshold": 0.01      # 1% ATR bajo
            },
            "bollinger": {
                "period": 20,              # Período para Bandas Bollinger
                "std_dev": 2,              # Desviaciones estándar
                "squeeze_threshold": 0.5,   # Umbral para squeeze
                "expansion_threshold": 2.5  # Umbral para expansión
            },
            "keltner": {
                "period": 20,              # Período para Keltner Channels
                "atr_multiplier": 2.0,     # Multiplicador de ATR
                "squeeze_threshold": 0.8    # Umbral para squeeze
            },
            "historical": {
                "short_term": 7,           # Días para volatilidad corto plazo
                "medium_term": 30,         # Días para volatilidad medio plazo
                "long_term": 90,           # Días para volatilidad largo plazo
                "percentile_high": 80,     # Percentil para volatilidad alta
                "percentile_low": 20       # Percentil para volatilidad baja
            }
        }

    def analyze(self, candlesticks: List[Dict]) -> Dict:
        """
        Realiza un análisis completo de volatilidad
        """
        try:
            if not self._validate_data(candlesticks):
                return self._get_default_analysis()

            # Calcular ATR y volatilidad histórica
            atr_analysis = self._analyze_atr(candlesticks)
            historical_volatility = self._calculate_historical_volatility(candlesticks)

            # Análisis de Bandas de Bollinger
            bollinger_analysis = self._analyze_bollinger_bands(candlesticks)

            # Análisis de Canales de Keltner
            keltner_analysis = self._analyze_keltner_channels(candlesticks)

            # Análisis de volatilidad por timeframes
            timeframe_analysis = self._analyze_timeframe_volatility(candlesticks)

            # Generar señales y condiciones
            signals = self._generate_volatility_signals(
                atr_analysis,
                historical_volatility,
                bollinger_analysis,
                keltner_analysis
            )

            return {
                'atr': atr_analysis,
                'historical': historical_volatility,
                'bollinger': bollinger_analysis,
                'keltner': keltner_analysis,
                'timeframes': timeframe_analysis,
                'current_state': self._determine_volatility_state(
                    atr_analysis,
                    historical_volatility,
                    bollinger_analysis,
                    keltner_analysis
                ),
                'signals': signals,
                'is_valid': True
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error en análisis de volatilidad: {str(e)}"))
            return self._get_default_analysis()

    def _analyze_atr(self, candlesticks: List[Dict]) -> Dict:
        """
        Analiza el Average True Range (ATR)
        """
        try:
            closes = [float(candle['close']) for candle in candlesticks]
            highs = [float(candle['high']) for candle in candlesticks]
            lows = [float(candle['low']) for candle in candlesticks]

            # Calcular True Ranges
            true_ranges = []
            for i in range(1, len(candlesticks)):
                true_range = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]),
                    abs(lows[i] - closes[i-1])
                )
                true_ranges.append(true_range)

            # Calcular ATR
            period = self.volatility_params["atr"]["period"]
            atr = sum(true_ranges[-period:]) / period if true_ranges else 0

            # Calcular ATR como porcentaje del precio
            current_price = closes[-1]
            atr_percent = (atr / current_price) * 100 if current_price > 0 else 0

            # Determinar nivel de ATR
            atr_level = self._determine_atr_level(atr_percent)

            # Calcular tendencia del ATR
            atr_trend = self._calculate_atr_trend(true_ranges)

            return {
                'value': atr,
                'percent': atr_percent,
                'level': atr_level,
                'trend': atr_trend,
                'is_high': atr_percent > self.volatility_params["atr"]["high_threshold"],
                'is_low': atr_percent < self.volatility_params["atr"]["low_threshold"]
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando ATR: {str(e)}"))
            return self._get_default_atr()

    def _analyze_bollinger_bands(self, candlesticks: List[Dict]) -> Dict:
        """
        Analiza las Bandas de Bollinger para volatilidad
        """
        try:
            closes = [float(candle['close']) for candle in candlesticks]
            period = self.volatility_params["bollinger"]["period"]
            std_dev = self.volatility_params["bollinger"]["std_dev"]

            if len(closes) < period:
                return self._get_default_bollinger()

            # Calcular SMA y bandas
            sma = sum(closes[-period:]) / period
            std = np.std(closes[-period:])

            upper = sma + (std_dev * std)
            lower = sma - (std_dev * std)

            # Calcular ancho de bandas
            bandwidth = (upper - lower) / sma

            # Detectar squeeze/expansión
            is_squeeze = bandwidth < self.volatility_params["bollinger"]["squeeze_threshold"]
            is_expansion = bandwidth > self.volatility_params["bollinger"]["expansion_threshold"]

            # Calcular %B
            current_price = closes[-1]
            percent_b = (current_price - lower) / (upper - lower) if upper != lower else 0.5

            return {
                'upper': upper,
                'middle': sma,
                'lower': lower,
                'bandwidth': bandwidth,
                'percent_b': percent_b,
                'is_squeeze': is_squeeze,
                'is_expansion': is_expansion,
                'trend': self._determine_bollinger_trend(closes, upper, lower)
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando Bandas de Bollinger: {str(e)}"))
            return self._get_default_bollinger()

    def _analyze_keltner_channels(self, candlesticks: List[Dict]) -> Dict:
        """
        Analiza los Canales de Keltner para volatilidad
        """
        try:
            closes = [float(candle['close']) for candle in candlesticks]
            highs = [float(candle['high']) for candle in candlesticks]
            lows = [float(candle['low']) for candle in candlesticks]

            period = self.volatility_params["keltner"]["period"]
            multiplier = self.volatility_params["keltner"]["atr_multiplier"]

            if len(closes) < period:
                return self._get_default_keltner()

            # Calcular EMA
            ema = self._calculate_ema(closes, period)

            # Calcular ATR
            atr = self._calculate_atr_value(highs, lows, closes)

            # Calcular bandas
            upper = ema + (multiplier * atr)
            lower = ema - (multiplier * atr)

            # Calcular ancho del canal
            channel_width = (upper - lower) / ema

            # Detectar squeeze
            is_squeeze = channel_width < self.volatility_params["keltner"]["squeeze_threshold"]

            current_price = closes[-1]
            return {
                'upper': upper,
                'middle': ema,
                'lower': lower,
                'channel_width': channel_width,
                'is_squeeze': is_squeeze,
                'position': self._determine_price_position(current_price, upper, lower)
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando Canales de Keltner: {str(e)}"))
            return self._get_default_keltner()

    def _calculate_historical_volatility(self, candlesticks: List[Dict]) -> Dict:
        """
        Calcula la volatilidad histórica en diferentes timeframes
        """
        try:
            closes = [float(candle['close']) for candle in candlesticks]

            # Calcular retornos logarítmicos
            returns = np.log(np.array(closes[1:]) / np.array(closes[:-1]))

            # Calcular volatilidad por timeframes
            volatilities = {
                'short_term': self._calculate_volatility_for_period(
                    returns,
                    self.volatility_params["historical"]["short_term"]
                ),
                'medium_term': self._calculate_volatility_for_period(
                    returns,
                    self.volatility_params["historical"]["medium_term"]
                ),
                'long_term': self._calculate_volatility_for_period(
                    returns,
                    self.volatility_params["historical"]["long_term"]
                )
            }

            # Calcular percentiles
            percentile_high = np.percentile(returns, self.volatility_params["historical"]["percentile_high"])
            percentile_low = np.percentile(returns, self.volatility_params["historical"]["percentile_low"])

            return {
                'current': volatilities['short_term'],
                'historical': volatilities,
                'percentiles': {
                    'high': percentile_high,
                    'low': percentile_low
                },
                'state': self._determine_volatility_state_from_history(volatilities)
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando volatilidad histórica: {str(e)}"))
            return self._get_default_historical()

    def _analyze_timeframe_volatility(self, candlesticks: List[Dict]) -> Dict:
        """
        Analiza la volatilidad en diferentes timeframes
        """
        try:
            # Agrupar datos por timeframe
            timeframes = {
                '1H': candlesticks[-24:],    # Último día
                '4H': candlesticks[-72:],    # Últimos 3 días
                '1D': candlesticks[-168:]    # Última semana
            }

            analysis = {}
            for timeframe, data in timeframes.items():
                if len(data) >= 2:
                    analysis[timeframe] = {
                        'volatility': self._calculate_timeframe_volatility(data),
                        'atr': self._calculate_atr_value(
                            [float(c['high']) for c in data],
                            [float(c['low']) for c in data],
                            [float(c['close']) for c in data]
                        ),
                        'range': self._calculate_price_range(data)
                    }

            return analysis

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando volatilidad por timeframes: {str(e)}"))
            return {}

    def _calculate_volatility_for_period(self, returns: np.ndarray, period: int) -> float:
        """
        Calcula la volatilidad para un período específico
        """
        if len(returns) < period:
            return 0.0

        return np.std(returns[-period:]) * np.sqrt(252)  # Anualizada

    def _calculate_timeframe_volatility(self, candlesticks: List[Dict]) -> float:
        """
        Calcula la volatilidad para un timeframe específico
        """
        try:
            if len(candlesticks) < 2:
                return 0.0

            closes = [float(candle['close']) for candle in candlesticks]
            returns = np.log(np.array(closes[1:]) / np.array(closes[:-1]))

            return np.std(returns) * np.sqrt(len(candlesticks))

        except Exception:
            return 0.0

    def _calculate_price_range(self, candlesticks: List[Dict]) -> Dict:
        """
        Calcula el rango de precios para un conjunto de velas
        """
        try:
            highs = [float(candle['high']) for candle in candlesticks]
            lows = [float(candle['low']) for candle in candlesticks]

            return {
                'high': max(highs),
                'low': min(lows),
                'range_percent': (max(highs) - min(lows)) / min(lows) * 100
            }

        except Exception:
            return {'high': 0, 'low': 0, 'range_percent': 0}

    def _calculate_ema(self, values: List[float], period: int) -> float:
        """
        Calcula EMA (Exponential Moving Average)
        """
        if not values or period <= 0:
            return 0.0

        multiplier = 2 / (period + 1)
        ema = values[0]

        for value in values[1:]:
            ema = value * multiplier + ema * (1 - multiplier)

        return ema

    def _calculate_atr_value(self, highs: List[float], lows: List[float],
                           closes: List[float]) -> float:
        """
        Calcula el valor del ATR
        """
        try:
            if len(highs) < 2:
                return 0.0

            true_ranges = []
            for i in range(1, len(highs)):
                true_ranges.append(
                    max(
                        highs[i] - lows[i],
                        abs(highs[i] - closes[i-1]),
                        abs(lows[i] - closes[i-1])
                    )
                )

            return sum(true_ranges) / len(true_ranges)

        except Exception:
            return 0.0

    def _determine_atr_level(self, atr_percent: float) -> str:
        """
        Determina el nivel de ATR
        """
        if atr_percent > self.volatility_params["atr"]["high_threshold"]:
            return 'high'
        elif atr_percent > self.volatility_params["atr"]["medium_threshold"]:
            return 'medium'
        return 'low'

    def _calculate_atr_trend(self, true_ranges: List[float]) -> str:
        """
        Calcula la tendencia del ATR
        """
        if len(true_ranges) < 2:
            return 'neutral'

        recent_avg = sum(true_ranges[-5:]) / 5
        previous_avg = sum(true_ranges[-10:-5]) / 5

        if recent_avg > previous_avg * 1.1:
            return 'increasing'
        elif recent_avg < previous_avg * 0.9:
            return 'decreasing'
        return 'neutral'

    def _determine_bollinger_trend(self, closes: List[float],
                                 upper: float, lower: float) -> str:
        """
        Determina la tendencia de las Bandas de Bollinger
        """
        if len(closes) < 2:
            return 'neutral'

        price = closes[-1]
        prev_price = closes[-2]

        if price > upper and prev_price > upper:
            return 'overbought'
        elif price < lower and prev_price < lower:
            return 'oversold'
        return 'neutral'

    def _determine_price_position(self, price: float,
                                upper: float, lower: float) -> str:
        """
        Determina la posición del precio en el canal
        """
        if price > upper:
            return 'above'
        elif price < lower:
            return 'below'
        return 'inside'

    def _determine_volatility_state(self, atr: Dict, historical: Dict,
                                  bollinger: Dict, keltner: Dict) -> Dict:
        """
        Determina el estado general de la volatilidad
        """
        try:
            # Contar indicadores que muestran alta volatilidad
            high_count = sum([
                atr['is_high'],
                historical['current'] > historical['percentiles']['high'],
                bollinger['is_expansion'],
                not keltner['is_squeeze']
            ])

            # Contar indicadores que muestran baja volatilidad
            low_count = sum([
                atr['is_low'],
                historical['current'] < historical['percentiles']['low'],
                bollinger['is_squeeze'],
                keltner['is_squeeze']
            ])

            # Determinar estado
            if high_count >= 3:
                state = 'high'
            elif low_count >= 3:
                state = 'low'
            else:
                state = 'normal'

            return {
                'state': state,
                'confidence': max(high_count, low_count) / 4,
                'indicators_high': high_count,
                'indicators_low': low_count
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error determinando estado de volatilidad: {str(e)}"))
            return {'state': 'normal', 'confidence': 0, 'indicators_high': 0, 'indicators_low': 0}

    def _generate_volatility_signals(self, atr: Dict, historical: Dict,
                                   bollinger: Dict, keltner: Dict) -> List[str]:
        """
        Genera señales basadas en el análisis de volatilidad
        """
        signals = []

        try:
            # Señales de ATR
            if atr['is_high']:
                signals.append(f"ATR alto ({atr['percent']:.1f}%) - Precaución")
            elif atr['is_low']:
                signals.append(f"ATR bajo ({atr['percent']:.1f}%) - Posible expansión")

            # Señales de Bollinger
            if bollinger['is_squeeze']:
                signals.append("Compresión de Bollinger - Posible ruptura")
            elif bollinger['is_expansion']:
                signals.append("Expansión de Bollinger - Alta volatilidad")

            # Señales de Keltner
            if keltner['is_squeeze']:
                signals.append("Compresión de Keltner - Posible ruptura")

            # Señales históricas
            current_vol = historical['current']
            if current_vol > historical['percentiles']['high']:
                signals.append(f"Volatilidad histórica alta ({current_vol:.1f}%)")
            elif current_vol < historical['percentiles']['low']:
                signals.append(f"Volatilidad histórica baja ({current_vol:.1f}%)")

            return signals

        except Exception as e:
            print(ConsoleColors.error(f"Error generando señales de volatilidad: {str(e)}"))
            return ["Error en análisis de volatilidad"]

    def _validate_data(self, candlesticks: List[Dict]) -> bool:
        """
        Valida que haya suficientes datos para el análisis
        """
        return bool(candlesticks and len(candlesticks) >= self.volatility_params["atr"]["period"])

    def _get_default_analysis(self) -> Dict:
        """
        Retorna análisis por defecto
        """
        return {
            'atr': self._get_default_atr(),
            'historical': self._get_default_historical(),
            'bollinger': self._get_default_bollinger(),
            'keltner': self._get_default_keltner(),
            'timeframes': {},
            'current_state': {'state': 'normal', 'confidence': 0},
            'signals': [],
            'is_valid': False
        }

    def _get_default_atr(self) -> Dict:
        return {
            'value': 0,
            'percent': 0,
            'level': 'low',
            'trend': 'neutral',
            'is_high': False,
            'is_low': True
        }

    def _get_default_bollinger(self) -> Dict:
        return {
            'upper': 0,
            'middle': 0,
            'lower': 0,
            'bandwidth': 0,
            'percent_b': 0.5,
            'is_squeeze': False,
            'is_expansion': False,
            'trend': 'neutral'
        }

    def _get_default_keltner(self) -> Dict:
        return {
            'upper': 0,
            'middle': 0,
            'lower': 0,
            'channel_width': 0,
            'is_squeeze': False,
            'position': 'inside'
        }

    def _get_default_historical(self) -> Dict:
        return {
            'current': 0,
            'historical': {
                'short_term': 0,
                'medium_term': 0,
                'long_term': 0
            },
            'percentiles': {
                'high': 0,
                'low': 0
            },
            'state': 'normal'
        }

from typing import Dict, List, Optional
import numpy as np
from utils.console_colors import ConsoleColors
from models.enums import MarketTrend

class TrendAnalyzer:
    def __init__(self):
        self.ema_periods = {
            "fast": 8,        # Señales rápidas
            "medium": 21,     # Medio plazo
            "slow": 50,       # Largo plazo
            "trend": 200,     # Tendencia principal
            "validation": [13, 34, 89]  # EMAs Fibonacci
        }

        self.trend_thresholds = {
            "strong": 0.6,    # Fuerza mínima para tendencia fuerte
            "moderate": 0.3,  # Fuerza mínima para tendencia moderada
            "min_separation": 0.5,  # % mínimo entre EMAs
            "alignment": 0.8  # % de EMAs que deben estar alineadas
        }

    def analyze_main_trend(self, market_data: Dict) -> Dict:
        """
        Analiza la tendencia principal usando múltiples timeframes.
        """
        try:
            trend_analysis = {
                'is_valid': False,
                'trend': 'neutral',
                'strength': 0,
                'data': {},
                'timeframes': {}
            }

            for timeframe, data in market_data.items():
                if not self._validate_data(data):
                    continue

                closes = self._extract_closes(data)
                if len(closes) < 50:
                    continue

                timeframe_trend = self._analyze_timeframe_trend(closes)
                trend_analysis['timeframes'][timeframe] = timeframe_trend
                trend_analysis['data'][timeframe] = closes

            # Analizar tendencia global
            if trend_analysis['timeframes']:
                trend_analysis.update(
                    self._determine_global_trend(trend_analysis['timeframes'])
                )

            return trend_analysis

        except Exception as e:
            print(ConsoleColors.error(f"Error en análisis de tendencia principal: {str(e)}"))
            return self._get_default_trend_analysis()

    def analyze_trend(self, candlesticks: List[Dict]) -> MarketTrend:
        """
        Analiza la tendencia del mercado usando EMAs y momentum
        """
        try:
            closes = self._extract_closes(candlesticks)

            # Calcular EMAs
            ema20 = self._calculate_ema(closes, 20)
            ema50 = self._calculate_ema(closes, 50)

            # Calcular momentum
            momentum = self._calculate_momentum(closes)

            # Determinar tendencia
            if ema20[-1] > ema50[-1] and momentum > 10:
                return MarketTrend.STRONG_UPTREND
            elif ema20[-1] > ema50[-1]:
                return MarketTrend.UPTREND
            elif ema20[-1] < ema50[-1] and momentum < -10:
                return MarketTrend.STRONG_DOWNTREND
            elif ema20[-1] < ema50[-1]:
                return MarketTrend.DOWNTREND

            return MarketTrend.NEUTRAL

        except Exception:
            return MarketTrend.NEUTRAL

    def _analyze_timeframe_trend(self, closes: List[float]) -> Dict:
        """Analiza la tendencia para un timeframe específico"""
        try:
            # Calcular EMAs
            ema20 = self._calculate_ema(closes, 20)[-1]
            ema50 = self._calculate_ema(closes, 50)[-1]
            ema200 = self._calculate_ema(closes, 200)[-1]

            # Calcular momentum
            momentum = self._calculate_momentum(closes)

            return {
                'price': closes[-1],
                'ema20': ema20,
                'ema50': ema50,
                'ema200': ema200,
                'momentum': momentum,
                'is_bullish': closes[-1] > ema20 > ema50 > ema200,
                'is_bearish': closes[-1] < ema20 < ema50 < ema200,
                'ema_alignment': self._calculate_ema_alignment(closes)
            }
        except Exception as e:
            print(ConsoleColors.error(f"Error analizando timeframe: {str(e)}"))
            return {}

    def _determine_global_trend(self, timeframe_trends: Dict) -> Dict:
        """Determina la tendencia global basada en múltiples timeframes"""
        try:
            timeframe_count = len(timeframe_trends)
            if timeframe_count == 0:
                return {'is_valid': False, 'trend': 'neutral', 'strength': 0}

            bullish_count = sum(1 for tf in timeframe_trends.values() if tf.get('is_bullish', False))
            bearish_count = sum(1 for tf in timeframe_trends.values() if tf.get('is_bearish', False))

            if bullish_count > timeframe_count / 2:
                trend = 'bullish'
                strength = bullish_count / timeframe_count
            elif bearish_count > timeframe_count / 2:
                trend = 'bearish'
                strength = bearish_count / timeframe_count
            else:
                trend = 'neutral'
                strength = 0.5

            return {
                'is_valid': True,
                'trend': trend,
                'strength': strength
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error determinando tendencia global: {str(e)}"))
            return {'is_valid': False, 'trend': 'neutral', 'strength': 0}

    def _calculate_ema(self, values: List[float], period: int) -> List[float]:
        """Calcula EMA (Exponential Moving Average)"""
        if not values:
            return []

        ema = [values[0]]
        multiplier = 2 / (period + 1)

        for price in values[1:]:
            ema.append((price * multiplier) + (ema[-1] * (1 - multiplier)))

        return ema

    def _calculate_momentum(self, closes: List[float], period: int = 20) -> float:
        """Calcula el momentum del precio"""
        try:
            if len(closes) < period:
                return 0.0
            return ((closes[-1] - closes[-period]) / closes[-period]) * 100
        except Exception:
            return 0.0

    def _calculate_ema_alignment(self, closes: List[float]) -> float:
        """Calcula el grado de alineación entre EMAs"""
        try:
            emas = []
            for period in self.ema_periods["validation"]:
                ema = self._calculate_ema(closes, period)
                if ema:
                    emas.append(ema[-1])

            if len(emas) < 2:
                return 0.0

            # Verificar que estén en orden (alcista o bajista)
            is_aligned = all(emas[i] > emas[i+1] for i in range(len(emas)-1))
            or_aligned = all(emas[i] < emas[i+1] for i in range(len(emas)-1))

            return 1.0 if is_aligned or or_aligned else 0.0

        except Exception:
            return 0.0

    def _extract_closes(self, candlesticks: List[Dict]) -> List[float]:
        """Extrae los precios de cierre de las velas"""
        closes = []
        for candle in candlesticks:
            try:
                if isinstance(candle, dict):
                    close = float(candle.get('close', 0))
                elif isinstance(candle, (list, tuple)) and len(candle) >= 4:
                    close = float(candle[4])
                else:
                    continue

                if close > 0:
                    closes.append(close)
            except (ValueError, TypeError, IndexError):
                continue
        return closes

    def _validate_data(self, data: List) -> bool:
        """Valida que los datos sean suficientes para el análisis"""
        return bool(data and len(data) >= 50)

    def _get_default_trend_analysis(self) -> Dict:
        """Retorna un análisis de tendencia por defecto"""
        return {
            'is_valid': False,
            'trend': 'neutral',
            'strength': 0,
            'data': {},
            'timeframes': {}
        }

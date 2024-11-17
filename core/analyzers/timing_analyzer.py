from typing import Dict, List, Optional, Tuple
import numpy as np
from utils.console_colors import ConsoleColors
from models.enums import EntryTiming
from models.data_classes import TimingWindow

class TimingAnalyzer:
    def __init__(self):
        self.timing_params = {
            "rsi": {
                "oversold": 30,
                "overbought": 70,
                "extreme_oversold": 20,
                "extreme_overbought": 80
            },
            "price_levels": {
                "support_proximity": 0.02,    # 2% cerca del soporte
                "resistance_proximity": 0.02,  # 2% cerca de la resistencia
                "breakout_confirmation": 0.01  # 1% para confirmar ruptura
            },
            "volume": {
                "spike_threshold": 2.0,       # 2x sobre promedio
                "significant_level": 1.5,     # 1.5x sobre promedio
                "confirmation_periods": 3      # Períodos para confirmar volumen
            },
            "volatility": {
                "high_threshold": 0.05,       # 5% volatilidad alta
                "low_threshold": 0.01,        # 1% volatilidad baja
                "adjustment_factor": 1.5       # Factor de ajuste timeframes
            },
            "momentum": {
                "strong_threshold": 0.1,      # 10% para momentum fuerte
                "weak_threshold": 0.05        # 5% para momentum débil
            },
            "timeframes": {
                "immediate": "0-4 horas",
                "short": "4-8 horas",
                "medium": "8-12 horas",
                "long": "12-24 horas"
            }
        }

    def analyze_entry_timing(self, candlesticks: List[Dict], current_price: float,
                           technical_analysis: Dict, support_resistance: Dict) -> TimingWindow:
        """
        Analiza el mejor momento para entrar al mercado
        """
        try:
            if not self._validate_data(candlesticks):
                return self._get_default_timing()

            # Análisis de condiciones técnicas
            rsi_analysis = self._analyze_rsi_conditions(technical_analysis.get('rsi', 50))

            # Análisis de niveles de precio
            price_analysis = self._analyze_price_levels(
                current_price,
                support_resistance
            )

            # Análisis de volatilidad
            volatility_analysis = self._analyze_volatility(candlesticks)

            # Análisis de volumen
            volume_analysis = self._analyze_volume_conditions(candlesticks)

            # Análisis de momentum
            momentum_analysis = self._analyze_momentum(candlesticks)

            # Determinar timing y timeframe
            timing, timeframe = self._determine_entry_timing(
                rsi_analysis,
                price_analysis,
                volatility_analysis,
                volume_analysis,
                momentum_analysis
            )

            # Calcular precio objetivo
            target_price = self._calculate_target_price(
                current_price,
                price_analysis,
                timing
            )

            # Calcular confianza
            confidence = self._calculate_timing_confidence(
                rsi_analysis,
                price_analysis,
                volatility_analysis,
                volume_analysis,
                momentum_analysis
            )

            # Generar condiciones
            conditions = self._generate_timing_conditions(
                rsi_analysis,
                price_analysis,
                volatility_analysis,
                volume_analysis,
                momentum_analysis
            )

            return TimingWindow(
                timing=timing,
                timeframe=timeframe,
                target_price=target_price,
                confidence=confidence,
                conditions=conditions
            )

        except Exception as e:
            print(ConsoleColors.error(f"Error en análisis de timing: {str(e)}"))
            return self._get_default_timing()

    def _analyze_rsi_conditions(self, rsi: float) -> Dict:
        """
        Analiza las condiciones del RSI para timing
        """
        try:
            conditions = {
                'is_oversold': False,
                'is_overbought': False,
                'is_extreme': False,
                'signal': 'neutral',
                'strength': 0.0
            }

            if rsi <= self.timing_params["rsi"]["extreme_oversold"]:
                conditions.update({
                    'is_oversold': True,
                    'is_extreme': True,
                    'signal': 'strong_buy',
                    'strength': 1.0
                })
            elif rsi <= self.timing_params["rsi"]["oversold"]:
                conditions.update({
                    'is_oversold': True,
                    'signal': 'buy',
                    'strength': 0.8
                })
            elif rsi >= self.timing_params["rsi"]["extreme_overbought"]:
                conditions.update({
                    'is_overbought': True,
                    'is_extreme': True,
                    'signal': 'strong_sell',
                    'strength': 1.0
                })
            elif rsi >= self.timing_params["rsi"]["overbought"]:
                conditions.update({
                    'is_overbought': True,
                    'signal': 'sell',
                    'strength': 0.8
                })

            conditions['value'] = rsi
            return conditions

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando RSI: {str(e)}"))
            return {
                'is_oversold': False,
                'is_overbought': False,
                'is_extreme': False,
                'signal': 'neutral',
                'strength': 0.0,
                'value': 50
            }

    def _analyze_price_levels(self, current_price: float, support_resistance: Dict) -> Dict:
        """
        Analiza la posición del precio respecto a niveles clave
        """
        try:
            analysis = {
                'near_support': False,
                'near_resistance': False,
                'price_position': 'neutral',
                'strength': 0.0,
                'closest_level': None,
                'distance_to_level': float('inf')
            }

            support = support_resistance.get('support', [])
            resistance = support_resistance.get('resistance', [])

            if support:
                closest_support = min(support, key=lambda x: abs(current_price - x))
                support_distance = (current_price - closest_support) / closest_support
                if abs(support_distance) <= self.timing_params["price_levels"]["support_proximity"]:
                    analysis.update({
                        'near_support': True,
                        'price_position': 'at_support',
                        'strength': 1 - (abs(support_distance) / self.timing_params["price_levels"]["support_proximity"]),
                        'closest_level': closest_support,
                        'distance_to_level': support_distance
                    })

            if resistance:
                closest_resistance = min(resistance, key=lambda x: abs(current_price - x))
                resistance_distance = (closest_resistance - current_price) / current_price
                if abs(resistance_distance) <= self.timing_params["price_levels"]["resistance_proximity"]:
                    analysis.update({
                        'near_resistance': True,
                        'price_position': 'at_resistance',
                        'strength': 1 - (abs(resistance_distance) / self.timing_params["price_levels"]["resistance_proximity"]),
                        'closest_level': closest_resistance,
                        'distance_to_level': resistance_distance
                    })

            return analysis

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando niveles de precio: {str(e)}"))
            return {
                'near_support': False,
                'near_resistance': False,
                'price_position': 'neutral',
                'strength': 0.0,
                'closest_level': None,
                'distance_to_level': float('inf')
            }

    def _analyze_volatility(self, candlesticks: List[Dict]) -> Dict:
        """
        Analiza la volatilidad para ajustar el timing
        """
        try:
            closes = [float(candle['close']) for candle in candlesticks]
            highs = [float(candle['high']) for candle in candlesticks]
            lows = [float(candle['low']) for candle in candlesticks]

            # Calcular True Range
            true_ranges = []
            for i in range(1, len(candlesticks)):
                true_range = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]),
                    abs(lows[i] - closes[i-1])
                )
                true_ranges.append(true_range)

            # Calcular ATR
            atr = sum(true_ranges[-14:]) / 14 if true_ranges else 0

            # Calcular volatilidad como % del precio
            current_price = closes[-1]
            volatility = atr / current_price if current_price > 0 else 0

            return {
                'value': volatility,
                'is_high': volatility > self.timing_params["volatility"]["high_threshold"],
                'is_low': volatility < self.timing_params["volatility"]["low_threshold"],
                'timeframe_adjustment': max(1, volatility * self.timing_params["volatility"]["adjustment_factor"])
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando volatilidad: {str(e)}"))
            return {
                'value': 0.0,
                'is_high': False,
                'is_low': True,
                'timeframe_adjustment': 1.0
            }

    def _analyze_volume_conditions(self, candlesticks: List[Dict]) -> Dict:
        """
        Analiza las condiciones de volumen para timing
        """
        try:
            volumes = [float(candle['volume']) for candle in candlesticks]
            avg_volume = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else volumes[-1]

            current_volume = volumes[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

            # Analizar tendencia de volumen
            recent_volume_avg = sum(volumes[-3:]) / 3 if len(volumes) >= 3 else current_volume
            volume_trend = recent_volume_avg / avg_volume if avg_volume > 0 else 1.0

            return {
                'ratio': volume_ratio,
                'is_spike': volume_ratio > self.timing_params["volume"]["spike_threshold"],
                'is_significant': volume_ratio > self.timing_params["volume"]["significant_level"],
                'trend': 'increasing' if volume_trend > 1 else 'decreasing',
                'strength': min(1.0, volume_ratio / self.timing_params["volume"]["spike_threshold"])
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando volumen: {str(e)}"))
            return {
                'ratio': 1.0,
                'is_spike': False,
                'is_significant': False,
                'trend': 'neutral',
                'strength': 0.0
            }

    def _analyze_momentum(self, candlesticks: List[Dict]) -> Dict:
        """
        Analiza el momentum para timing
        """
        try:
            closes = [float(candle['close']) for candle in candlesticks]

            # Calcular cambio porcentual
            momentum = ((closes[-1] - closes[-20]) / closes[-20]) * 100 if len(closes) >= 20 else 0

            # Calcular aceleración
            recent_changes = [
                ((closes[i] - closes[i-1]) / closes[i-1]) * 100
                for i in range(max(1, len(closes)-5), len(closes))
            ]

            acceleration = sum(recent_changes) / len(recent_changes) if recent_changes else 0

            return {
                'value': momentum,
                'acceleration': acceleration,
                'is_strong': abs(momentum) > self.timing_params["momentum"]["strong_threshold"],
                'is_weak': abs(momentum) < self.timing_params["momentum"]["weak_threshold"],
                'direction': 'bullish' if momentum > 0 else 'bearish',
                'strength': min(1.0, abs(momentum) / self.timing_params["momentum"]["strong_threshold"])
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando momentum: {str(e)}"))
            return {
                'value': 0.0,
                'acceleration': 0.0,
                'is_strong': False,
                'is_weak': True,
                'direction': 'neutral',
                'strength': 0.0
            }

    def _determine_entry_timing(self, rsi_analysis: Dict, price_analysis: Dict,
                              volatility_analysis: Dict, volume_analysis: Dict,
                              momentum_analysis: Dict) -> Tuple[EntryTiming, str]:
        """
        Determina el timing de entrada y timeframe
        """
        try:
            # Entrada inmediata
            if (rsi_analysis['is_oversold'] or price_analysis['near_support']) and \
               (volume_analysis['is_significant'] or momentum_analysis['is_strong']):
                return EntryTiming.IMMEDIATE, self._adjust_timeframe(
                    self.timing_params["timeframes"]["immediate"],
                    volatility_analysis['timeframe_adjustment']
                )

            # Esperar retroceso
            if rsi_analysis['is_overbought'] or price_analysis['near_resistance']:
                return EntryTiming.WAIT_DIP, self._adjust_timeframe(
                    self.timing_params["timeframes"]["long"],
                    volatility_analysis['timeframe_adjustment']
                )

            # Esperar ruptura
            if price_analysis['near_resistance'] and \
               momentum_analysis['direction'] == 'bullish' and \
               volume_analysis['is_significant']:
                return EntryTiming.WAIT_BREAKOUT, self._adjust_timeframe(
                    self.timing_params["timeframes"]["short"],
                    volatility_analysis['timeframe_adjustment']
                )

            # Esperar consolidación
            if volatility_analysis['is_high'] or \
               (not volume_analysis['is_significant'] and momentum_analysis['is_weak']):
                return EntryTiming.WAIT_CONSOLIDATION, self._adjust_timeframe(
                    self.timing_params["timeframes"]["medium"],
                    volatility_analysis['timeframe_adjustment']
                )

            # Por defecto
            return EntryTiming.NOT_RECOMMENDED, "N/A"

        except Exception as e:
            print(ConsoleColors.error(f"Error determinando timing: {str(e)}"))
            return EntryTiming.NOT_RECOMMENDED, "N/A"

    def _calculate_target_price(self, current_price: float, price_analysis: Dict,
                              timing: EntryTiming) -> Optional[float]:
        """
        Calcula el precio objetivo basado en el timing
        """
        try:
            if timing == EntryTiming.IMMEDIATE:
                return current_price

            if timing == EntryTiming.WAIT_DIP and price_analysis['near_resistance']:
                return price_analysis['closest_level'] * 0.95  # 5% bajo resistencia

            if timing == EntryTiming.WAIT_BREAKOUT and price_analysis['near_resistance']:
                return price_analysis['closest_level'] * 1.02  # 2% sobre resistencia

            if timing == EntryTiming.WAIT_CONSOLIDATION:
                return current_price * 0.98  # 2% bajo precio actual

            return current_price

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando precio objetivo: {str(e)}"))
            return current_price

    def _calculate_timing_confidence(self, rsi_analysis: Dict,
                                  price_analysis: Dict,
                                  volatility_analysis: Dict,
                                  volume_analysis: Dict,
                                  momentum_analysis: Dict) -> float:
        """
        Calcula la confianza en el timing actual
        """
        try:
            confidence_factors = []

            # RSI factor (30%)
            if rsi_analysis['is_extreme']:
                confidence_factors.append(0.3)
            elif rsi_analysis['is_oversold'] or rsi_analysis['is_overbought']:
                confidence_factors.append(0.2)
            else:
                confidence_factors.append(0.1)

            # Price levels factor (25%)
            if price_analysis['near_support'] or price_analysis['near_resistance']:
                confidence_factors.append(0.25 * price_analysis['strength'])
            else:
                confidence_factors.append(0.1)

            # Volume factor (20%)
            if volume_analysis['is_spike']:
                confidence_factors.append(0.2)
            elif volume_analysis['is_significant']:
                confidence_factors.append(0.15)
            else:
                confidence_factors.append(0.05)

            # Momentum factor (15%)
            if momentum_analysis['is_strong']:
                confidence_factors.append(0.15)
            elif not momentum_analysis['is_weak']:
                confidence_factors.append(0.1)
            else:
                confidence_factors.append(0.05)

            # Volatility factor (10%)
            if not volatility_analysis['is_high']:
                confidence_factors.append(0.1)
            else:
                confidence_factors.append(0.05)

            return min(1.0, sum(confidence_factors))

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando confianza: {str(e)}"))
            return 0.0

    def _generate_timing_conditions(self, rsi_analysis: Dict,
                                 price_analysis: Dict,
                                 volatility_analysis: Dict,
                                 volume_analysis: Dict,
                                 momentum_analysis: Dict) -> List[str]:
        """
        Genera lista de condiciones que justifican el timing
        """
        try:
            conditions = []

            # Condiciones de RSI
            if rsi_analysis['is_oversold']:
                conditions.append(f"RSI en sobreventa ({rsi_analysis['value']:.1f})")
            elif rsi_analysis['is_overbought']:
                conditions.append(f"RSI en sobrecompra ({rsi_analysis['value']:.1f})")

            # Condiciones de precio
            if price_analysis['near_support']:
                conditions.append(f"Precio cerca del soporte (${price_analysis['closest_level']:.8f})")
            elif price_analysis['near_resistance']:
                conditions.append(f"Precio cerca de resistencia (${price_analysis['closest_level']:.8f})")

            # Condiciones de volatilidad
            if volatility_analysis['is_high']:
                conditions.append(f"Alta volatilidad ({volatility_analysis['value']:.1%})")
            else:
                conditions.append(f"Baja volatilidad ({volatility_analysis['value']:.1%})")

            # Condiciones de volumen
            if volume_analysis['is_spike']:
                conditions.append(f"Pico de volumen ({volume_analysis['ratio']:.1f}x promedio)")
            elif volume_analysis['is_significant']:
                conditions.append(f"Volumen significativo ({volume_analysis['ratio']:.1f}x promedio)")

            # Condiciones de momentum
            if momentum_analysis['is_strong']:
                conditions.append(
                    f"Momentum fuerte {momentum_analysis['direction']} "
                    f"({momentum_analysis['value']:.1f}%)"
                )

            return conditions

        except Exception as e:
            print(ConsoleColors.error(f"Error generando condiciones: {str(e)}"))
            return ["Error analizando condiciones"]

    def _adjust_timeframe(self, timeframe: str, adjustment_factor: float) -> str:
        """
        Ajusta el timeframe según la volatilidad
        """
        try:
            if timeframe == "N/A":
                return timeframe

            # Extraer números del timeframe
            import re
            numbers = re.findall(r'\d+', timeframe)
            if len(numbers) != 2:
                return timeframe

            # Ajustar valores
            min_time = int(float(numbers[0]) * adjustment_factor)
            max_time = int(float(numbers[1]) * adjustment_factor)

            return f"{min_time}-{max_time} horas"

        except Exception as e:
            print(ConsoleColors.error(f"Error ajustando timeframe: {str(e)}"))
            return timeframe

    def _validate_data(self, candlesticks: List[Dict]) -> bool:
        """
        Valida que haya suficientes datos para el análisis
        """
        return bool(candlesticks and len(candlesticks) >= 20)

    def _get_default_timing(self) -> TimingWindow:
        """
        Retorna una ventana de timing por defecto
        """
        return TimingWindow(
            timing=EntryTiming.NOT_RECOMMENDED,
            timeframe="N/A",
            target_price=None,
            confidence=0.0,
            conditions=["Datos insuficientes para análisis"]
        )

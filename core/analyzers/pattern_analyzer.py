from typing import Dict, List, Optional
import numpy as np
from utils.console_colors import ConsoleColors

class PatternAnalyzer:
    def __init__(self):
        self.pattern_params = {
            "min_pattern_size": 0.01,  # Tamaño mínimo para considerar un patrón
            "confirmation_candles": 2,  # Velas necesarias para confirmar
            "reliability_thresholds": {
                "high": 0.8,
                "moderate": 0.6,
                "low": 0.4
            },
            "body_to_wick_ratio": {
                "doji": 0.1,        # Ratio máximo para considerar doji
                "hammer": 0.3,      # Ratio máximo cuerpo para martillo
                "marubozu": 0.9     # Ratio mínimo cuerpo para marubozu
            }
        }

    def analyze(self, candlesticks: List[Dict]) -> Dict:
        """
        Analiza los patrones de velas en los datos proporcionados
        """
        try:
            if not self._validate_data(candlesticks):
                return self._get_default_analysis()

            # Detectar patrones individuales
            single_patterns = self._identify_single_patterns(candlesticks[-5:])

            # Detectar patrones múltiples
            multiple_patterns = self._identify_multiple_patterns(candlesticks[-5:])

            # Consolidar todos los patrones encontrados
            all_patterns = single_patterns + multiple_patterns

            # Calcular señal dominante
            signal = self._calculate_dominant_signal(all_patterns)

            return {
                'patterns': all_patterns,
                'pattern_count': len(all_patterns),
                'signal': signal,
                'reliability': self._calculate_overall_reliability(all_patterns),
                'has_confirmation': self._check_pattern_confirmation(candlesticks, all_patterns),
                'latest_pattern': all_patterns[-1] if all_patterns else None
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error en análisis de patrones: {str(e)}"))
            return self._get_default_analysis()

    def _identify_single_patterns(self, candles: List[Dict]) -> List[Dict]:
        """Identifica patrones de vela única"""
        patterns = []

        for i, candle in enumerate(candles):
            # Extraer datos de la vela
            data = self._extract_candle_data(candle)
            if not data:
                continue

            # Doji
            if self._is_doji(data):
                patterns.append({
                    'type': 'neutral',
                    'name': 'Doji',
                    'position': i,
                    'reliability': 0.6,
                    'significance': self._calculate_pattern_significance(data)
                })

            # Martillo
            if self._is_hammer(data):
                patterns.append({
                    'type': 'bullish',
                    'name': 'Hammer',
                    'position': i,
                    'reliability': 0.7,
                    'significance': self._calculate_pattern_significance(data)
                })

            # Estrella fugaz
            if self._is_shooting_star(data):
                patterns.append({
                    'type': 'bearish',
                    'name': 'Shooting Star',
                    'position': i,
                    'reliability': 0.7,
                    'significance': self._calculate_pattern_significance(data)
                })

            # Marubozu
            if self._is_marubozu(data):
                pattern_type = 'bullish' if data['close'] > data['open'] else 'bearish'
                patterns.append({
                    'type': pattern_type,
                    'name': f"{pattern_type.capitalize()} Marubozu",
                    'position': i,
                    'reliability': 0.8,
                    'significance': self._calculate_pattern_significance(data)
                })

        return patterns

    def _identify_multiple_patterns(self, candles: List[Dict]) -> List[Dict]:
        """Identifica patrones de múltiples velas"""
        patterns = []

        if len(candles) < 3:
            return patterns

        # Patrón envolvente alcista
        if self._is_bullish_engulfing(candles[-2:]):
            patterns.append({
                'type': 'bullish',
                'name': 'Bullish Engulfing',
                'position': -2,
                'reliability': 0.8,
                'significance': self._calculate_multiple_pattern_significance(candles[-2:])
            })

        # Patrón envolvente bajista
        if self._is_bearish_engulfing(candles[-2:]):
            patterns.append({
                'type': 'bearish',
                'name': 'Bearish Engulfing',
                'position': -2,
                'reliability': 0.8,
                'significance': self._calculate_multiple_pattern_significance(candles[-2:])
            })

        # Three White Soldiers
        if self._is_three_white_soldiers(candles[-3:]):
            patterns.append({
                'type': 'bullish',
                'name': 'Three White Soldiers',
                'position': -3,
                'reliability': 0.9,
                'significance': self._calculate_multiple_pattern_significance(candles[-3:])
            })

        # Three Black Crows
        if self._is_three_black_crows(candles[-3:]):
            patterns.append({
                'type': 'bearish',
                'name': 'Three Black Crows',
                'position': -3,
                'reliability': 0.9,
                'significance': self._calculate_multiple_pattern_significance(candles[-3:])
            })

        # Morning Star
        if self._is_morning_star(candles[-3:]):
            patterns.append({
                'type': 'bullish',
                'name': 'Morning Star',
                'position': -3,
                'reliability': 0.85,
                'significance': self._calculate_multiple_pattern_significance(candles[-3:])
            })

        # Evening Star
        if self._is_evening_star(candles[-3:]):
            patterns.append({
                'type': 'bearish',
                'name': 'Evening Star',
                'position': -3,
                'reliability': 0.85,
                'significance': self._calculate_multiple_pattern_significance(candles[-3:])
            })

        return patterns

    def _is_doji(self, data: Dict) -> bool:
        """Verifica si la vela es un doji"""
        body_size = abs(data['close'] - data['open'])
        total_size = data['high'] - data['low']

        if total_size == 0:
            return False

        body_ratio = body_size / total_size
        return body_ratio <= self.pattern_params["body_to_wick_ratio"]["doji"]

    def _is_hammer(self, data: Dict) -> bool:
        """Verifica si la vela es un martillo"""
        body_size = abs(data['close'] - data['open'])
        lower_wick = min(data['open'], data['close']) - data['low']
        upper_wick = data['high'] - max(data['open'], data['close'])

        if body_size == 0:
            return False

        return (lower_wick > body_size * 2 and
                upper_wick < body_size * 0.5 and
                body_size > self.pattern_params["min_pattern_size"])

    def _is_shooting_star(self, data: Dict) -> bool:
        """Verifica si la vela es una estrella fugaz"""
        body_size = abs(data['close'] - data['open'])
        lower_wick = min(data['open'], data['close']) - data['low']
        upper_wick = data['high'] - max(data['open'], data['close'])

        if body_size == 0:
            return False

        return (upper_wick > body_size * 2 and
                lower_wick < body_size * 0.5 and
                body_size > self.pattern_params["min_pattern_size"])

    def _is_marubozu(self, data: Dict) -> bool:
        """Verifica si la vela es un marubozu"""
        body_size = abs(data['close'] - data['open'])
        total_size = data['high'] - data['low']

        if total_size == 0:
            return False

        body_ratio = body_size / total_size
        return body_ratio >= self.pattern_params["body_to_wick_ratio"]["marubozu"]

    def _is_bullish_engulfing(self, candles: List[Dict]) -> bool:
        """Verifica si hay un patrón envolvente alcista"""
        if len(candles) < 2:
            return False

        prev = self._extract_candle_data(candles[-2])
        curr = self._extract_candle_data(candles[-1])

        if not prev or not curr:
            return False

        return (prev['close'] < prev['open'] and  # Vela previa bajista
                curr['close'] > curr['open'] and  # Vela actual alcista
                curr['open'] < prev['close'] and  # Abre por debajo
                curr['close'] > prev['open'])     # Cierra por encima

    def _is_bearish_engulfing(self, candles: List[Dict]) -> bool:
        """Verifica si hay un patrón envolvente bajista"""
        if len(candles) < 2:
            return False

        prev = self._extract_candle_data(candles[-2])
        curr = self._extract_candle_data(candles[-1])

        if not prev or not curr:
            return False

        return (prev['close'] > prev['open'] and  # Vela previa alcista
                curr['close'] < curr['open'] and  # Vela actual bajista
                curr['open'] > prev['close'] and  # Abre por encima
                curr['close'] < prev['open'])     # Cierra por debajo

    def _is_three_white_soldiers(self, candles: List[Dict]) -> bool:
        """Verifica el patrón Three White Soldiers"""
        if len(candles) < 3:
            return False

        data = [self._extract_candle_data(candle) for candle in candles]
        if not all(data):
            return False

        # Verificar tres velas alcistas consecutivas
        is_bullish = all(d['close'] > d['open'] for d in data)
        # Cada vela abre dentro del cuerpo de la anterior
        opens_ok = all(data[i]['open'] > data[i-1]['open'] for i in range(1, 3))
        # Cada vela cierra más arriba que la anterior
        closes_ok = all(data[i]['close'] > data[i-1]['close'] for i in range(1, 3))

        return is_bullish and opens_ok and closes_ok

    def _is_three_black_crows(self, candles: List[Dict]) -> bool:
        """Verifica el patrón Three Black Crows"""
        if len(candles) < 3:
            return False

        data = [self._extract_candle_data(candle) for candle in candles]
        if not all(data):
            return False

        # Verificar tres velas bajistas consecutivas
        is_bearish = all(d['close'] < d['open'] for d in data)
        # Cada vela abre dentro del cuerpo de la anterior
        opens_ok = all(data[i]['open'] < data[i-1]['open'] for i in range(1, 3))
        # Cada vela cierra más abajo que la anterior
        closes_ok = all(data[i]['close'] < data[i-1]['close'] for i in range(1, 3))

        return is_bearish and opens_ok and closes_ok

    def _is_morning_star(self, candles: List[Dict]) -> bool:
        """Verifica el patrón Morning Star"""
        if len(candles) < 3:
            return False

        data = [self._extract_candle_data(candle) for candle in candles]
        if not all(data):
            return False

        # Primera vela bajista
        first_bearish = data[0]['close'] < data[0]['open']
        # Segunda vela pequeña (doji o similar)
        second_small = abs(data[1]['close'] - data[1]['open']) < abs(data[0]['close'] - data[0]['open']) * 0.3
        # Tercera vela alcista
        third_bullish = data[2]['close'] > data[2]['open']
        # Gap entre primera y segunda
        gap_down = data[1]['high'] < data[0]['close']
        # Gap entre segunda y tercera
        gap_up = data[1]['low'] > data[2]['open']

        return first_bearish and second_small and third_bullish and gap_down and gap_up

    def _is_evening_star(self, candles: List[Dict]) -> bool:
        """Verifica el patrón Evening Star"""
        if len(candles) < 3:
            return False

        data = [self._extract_candle_data(candle) for candle in candles]
        if not all(data):
            return False

        # Primera vela alcista
        first_bullish = data[0]['close'] > data[0]['open']
        # Segunda vela pequeña (doji o similar)
        second_small = abs(data[1]['close'] - data[1]['open']) < abs(data[0]['close'] - data[0]['open']) * 0.3
        # Tercera vela bajista
        third_bearish = data[2]['close'] < data[2]['open']
        # Gap entre primera y segunda
        gap_up = data[1]['low'] > data[0]['close']
        # Gap entre segunda y tercera
        gap_down = data[1]['high'] < data[2]['open']

        return first_bullish and second_small and third_bearish and gap_up and gap_down

    def _calculate_pattern_significance(self, data: Dict) -> float:
        """Calcula la significancia de un patrón de vela única"""
        body_size = abs(data['close'] - data['open'])
        total_size = data['high'] - data['low']

        if total_size == 0:
            return 0.0

        # Considerar tamaño relativo y posición en el rango
        position_score = abs((data['close'] - data['low']) / total_size - 0.5) * 2
        size_score = total_size / data['open']  # Tamaño relativo al precio

        return min(1.0, (position_score + size_score) / 2)

    def _calculate_multiple_pattern_significance(self, candles: List[Dict]) -> float:
        """Calcula la significancia de un patrón de múltiples velas"""
        significances = []

        for candle in candles:
            data = self._extract_candle_data(candle)
            if data:
                significances.append(self._calculate_pattern_significance(data))

        return sum(significances) / len(significances) if significances else 0.0


    def _calculate_dominant_signal(self, patterns: List[Dict]) -> Dict:
        """
        Calcula la señal dominante basada en los patrones encontrados
        """
        if not patterns:
            return {
                'direction': 'neutral',
                'strength': 0.0,
                'confidence': 0.0,
                'reasoning': ['No se encontraron patrones significativos']
            }

        # Calcular peso de las señales
        bullish_weight = 0.0
        bearish_weight = 0.0
        reasoning = []

        for pattern in patterns:
            weight = pattern['reliability'] * pattern['significance']

            if pattern['type'] == 'bullish':
                bullish_weight += weight
                reasoning.append(f"Patrón alcista: {pattern['name']} "
                               f"(Confiabilidad: {pattern['reliability']:.1%})")
            elif pattern['type'] == 'bearish':
                bearish_weight += weight
                reasoning.append(f"Patrón bajista: {pattern['name']} "
                               f"(Confiabilidad: {pattern['reliability']:.1%})")

        total_weight = bullish_weight + bearish_weight
        if total_weight == 0:
            return {
                'direction': 'neutral',
                'strength': 0.0,
                'confidence': 0.0,
                'reasoning': ['Patrones identificados sin peso significativo']
            }

        # Determinar dirección y fuerza
        if bullish_weight > bearish_weight:
            direction = 'bullish'
            strength = bullish_weight / total_weight
        elif bearish_weight > bullish_weight:
            direction = 'bearish'
            strength = bearish_weight / total_weight
        else:
            direction = 'neutral'
            strength = 0.0

        # Calcular confianza basada en el número y calidad de patrones
        confidence = self._calculate_pattern_confidence(patterns)

        return {
            'direction': direction,
            'strength': strength,
            'confidence': confidence,
            'reasoning': reasoning
        }

    def _calculate_overall_reliability(self, patterns: List[Dict]) -> float:
        """
        Calcula la confiabilidad general de los patrones encontrados
        """
        if not patterns:
            return 0.0

        weighted_reliability = sum(p['reliability'] * p['significance'] for p in patterns)
        return weighted_reliability / len(patterns)

    def _calculate_pattern_confidence(self, patterns: List[Dict]) -> float:
        """
        Calcula la confianza general en los patrones identificados
        """
        if not patterns:
            return 0.0

        pattern_count = len(patterns)
        avg_reliability = sum(p['reliability'] for p in patterns) / pattern_count
        avg_significance = sum(p['significance'] for p in patterns) / pattern_count
        direction_agreement = self._calculate_direction_agreement(patterns)

        confidence = (
            avg_reliability * 0.4 +      # 40% basado en confiabilidad
            avg_significance * 0.3 +     # 30% basado en significancia
            direction_agreement * 0.3     # 30% basado en acuerdo direccional
        )

        return min(1.0, confidence)

    def _calculate_direction_agreement(self, patterns: List[Dict]) -> float:
        """
        Calcula el nivel de acuerdo direccional entre los patrones
        """
        if not patterns:
            return 0.0

        bullish_count = sum(1 for p in patterns if p['type'] == 'bullish')
        bearish_count = sum(1 for p in patterns if p['type'] == 'bearish')
        total_count = len(patterns)

        max_agreement = max(bullish_count, bearish_count)
        return max_agreement / total_count if total_count > 0 else 0.0

    def _check_pattern_confirmation(self, candlesticks: List[Dict], patterns: List[Dict]) -> bool:
        """
        Verifica si los patrones están confirmados por el movimiento posterior
        """
        if not patterns or len(candlesticks) < self.pattern_params["confirmation_candles"]:
            return False

        latest_pattern = patterns[-1]
        pattern_position = latest_pattern['position']
        confirmation_candles = candlesticks[pattern_position + 1:]

        if not confirmation_candles:
            return False

        if latest_pattern['type'] == 'bullish':
            return self._confirm_bullish_pattern(confirmation_candles)
        elif latest_pattern['type'] == 'bearish':
            return self._confirm_bearish_pattern(confirmation_candles)

        return False

    def _confirm_bullish_pattern(self, confirmation_candles: List[Dict]) -> bool:
        """
        Confirma un patrón alcista
        """
        data = [self._extract_candle_data(candle) for candle in confirmation_candles]
        if not all(data):
            return False

        closes = [d['close'] for d in data]
        return closes[-1] > closes[0]

    def _confirm_bearish_pattern(self, confirmation_candles: List[Dict]) -> bool:
        """
        Confirma un patrón bajista
        """
        data = [self._extract_candle_data(candle) for candle in confirmation_candles]
        if not all(data):
            return False

        closes = [d['close'] for d in data]
        return closes[-1] < closes[0]

    def _extract_candle_data(self, candle: Dict) -> Optional[Dict]:
        """
        Extrae los datos relevantes de una vela
        """
        try:
            if isinstance(candle, dict):
                return {
                    'open': float(candle['open']),
                    'high': float(candle['high']),
                    'low': float(candle['low']),
                    'close': float(candle['close']),
                    'volume': float(candle.get('volume', 0))
                }
            elif isinstance(candle, (list, tuple)) and len(candle) >= 6:
                return {
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5])
                }
        except (ValueError, IndexError, KeyError, TypeError) as e:
            print(ConsoleColors.error(f"Error extrayendo datos de vela: {str(e)}"))
            return None

        return None

    def _validate_data(self, candlesticks: List[Dict]) -> bool:
        """
        Valida que haya suficientes datos para el análisis
        """
        return bool(candlesticks and len(candlesticks) >= 5)

    def _get_default_analysis(self) -> Dict:
        """
        Retorna un análisis por defecto cuando hay error o datos insuficientes
        """
        return {
            'patterns': [],
            'pattern_count': 0,
            'signal': {
                'direction': 'neutral',
                'strength': 0.0,
                'confidence': 0.0,
                'reasoning': ['Datos insuficientes para análisis']
            },
            'reliability': 0.0,
            'has_confirmation': False,
            'latest_pattern': None
        }

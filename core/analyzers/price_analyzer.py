from typing import Dict, List, Optional, Tuple
import numpy as np
from utils.console_colors import ConsoleColors

class PriceAnalyzer:
    def __init__(self):
        self.price_params = {
            "pivot_periods": 20,          # Períodos para análisis de pivotes
            "level_strength": 3,          # Toques necesarios para confirmar nivel
            "zone_thickness": 0.002,      # Grosor de zonas S/R (0.2%)
            "level_proximity": 0.01,      # Proximidad a niveles (1%)
            "structure_params": {
                "swing_threshold": 0.01,  # % para identificar swings
                "trend_validation": 3,    # Pivotes para confirmar tendencia
                "structure_break": 0.005  # % para ruptura de estructura
            }
        }

    def analyze(self, candlesticks: List[Dict]) -> Dict:
        """
        Realiza un análisis completo de precios, niveles y estructura de mercado
        """
        try:
            if not self._validate_data(candlesticks):
                return self._get_default_analysis()

            # Análisis de soporte/resistencia
            support_resistance = self._calculate_support_resistance(candlesticks)

            # Análisis de estructura de mercado
            market_structure = self._analyze_market_structure(candlesticks)

            # Identificar zonas de interés
            key_zones = self._identify_key_zones(candlesticks, support_resistance)

            # Análisis de precio actual
            current_analysis = self._analyze_current_price(
                candlesticks[-1],
                support_resistance,
                market_structure
            )

            return {
                'support_resistance': support_resistance,
                'market_structure': market_structure,
                'key_zones': key_zones,
                'current_analysis': current_analysis,
                'is_valid': True,
                'signals': self._generate_price_signals(
                    support_resistance,
                    market_structure,
                    current_analysis
                )
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error en análisis de precios: {str(e)}"))
            return self._get_default_analysis()

    def _calculate_support_resistance(self, candlesticks: List[Dict]) -> Dict:
        """
        Calcula niveles de soporte y resistencia
        """
        try:
            highs = [float(candle['high']) for candle in candlesticks]
            lows = [float(candle['low']) for candle in candlesticks]
            closes = [float(candle['close']) for candle in candlesticks]

            # Identificar pivotes
            pivot_highs = self._find_pivot_points(highs, 'high')
            pivot_lows = self._find_pivot_points(lows, 'low')

            # Agrupar niveles cercanos
            resistance_levels = self._cluster_levels(pivot_highs)
            support_levels = self._cluster_levels(pivot_lows)

            # Validar fuerza de niveles
            validated_resistances = self._validate_levels(resistance_levels, highs, 'resistance')
            validated_supports = self._validate_levels(support_levels, lows, 'support')

            # Calcular niveles dinámicos
            dynamic_levels = self._calculate_dynamic_levels(closes)

            return {
                'resistance': self._select_most_relevant_levels(validated_resistances),
                'support': self._select_most_relevant_levels(validated_supports),
                'dynamic': dynamic_levels,
                'zones': self._identify_sr_zones(validated_supports, validated_resistances)
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando S/R: {str(e)}"))
            return {'support': [], 'resistance': [], 'dynamic': [], 'zones': []}

    def _analyze_market_structure(self, candlesticks: List[Dict]) -> Dict:
        """
        Analiza la estructura del mercado
        """
        try:
            closes = [float(candle['close']) for candle in candlesticks]
            highs = [float(candle['high']) for candle in candlesticks]
            lows = [float(candle['low']) for candle in candlesticks]

            # Identificar swings
            swing_highs = self._find_swing_highs(highs)
            swing_lows = self._find_swing_lows(lows)

            # Analizar estructura
            structure = self._analyze_price_structure(
                swing_highs,
                swing_lows,
                closes
            )

            # Identificar niveles de quiebre
            breakout_levels = self._identify_breakout_levels(
                closes,
                structure['trend'],
                structure['key_levels']
            )

            return {
                'structure': structure,
                'breakouts': breakout_levels,
                'swings': {
                    'highs': swing_highs,
                    'lows': swing_lows
                },
                'trend': self._determine_structure_trend(structure)
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando estructura: {str(e)}"))
            return self._get_default_structure()

    def _identify_key_zones(self, candlesticks: List[Dict], support_resistance: Dict) -> List[Dict]:
        """
        Identifica zonas clave de precio
        """
        try:
            zones = []

            # Combinar soportes y resistencias
            all_levels = (
                [(level, 'support') for level in support_resistance['support']] +
                [(level, 'resistance') for level in support_resistance['resistance']]
            )

            # Ordenar niveles por precio
            sorted_levels = sorted(all_levels, key=lambda x: x[0])

            # Identificar zonas de confluencia
            current_zone = None
            for level, level_type in sorted_levels:
                if current_zone is None:
                    current_zone = {
                        'start': level * (1 - self.price_params["zone_thickness"]),
                        'end': level * (1 + self.price_params["zone_thickness"]),
                        'types': [level_type],
                        'strength': 1
                    }
                else:
                    # Si el nivel está dentro de la zona actual
                    if level <= current_zone['end'] * (1 + self.price_params["zone_thickness"]):
                        current_zone['end'] = level * (1 + self.price_params["zone_thickness"])
                        current_zone['types'].append(level_type)
                        current_zone['strength'] += 1
                    else:
                        # Guardar zona actual y comenzar nueva
                        zones.append(current_zone)
                        current_zone = {
                            'start': level * (1 - self.price_params["zone_thickness"]),
                            'end': level * (1 + self.price_params["zone_thickness"]),
                            'types': [level_type],
                            'strength': 1
                        }

            # Añadir última zona
            if current_zone:
                zones.append(current_zone)

            return zones

        except Exception as e:
            print(ConsoleColors.error(f"Error identificando zonas clave: {str(e)}"))
            return []

    def _analyze_current_price(self, current_candle: Dict,
                             support_resistance: Dict,
                             market_structure: Dict) -> Dict:
        """
        Analiza la posición actual del precio
        """
        try:
            current_price = float(current_candle['close'])

            # Encontrar nivel más cercano
            closest_support = self._find_closest_level(current_price, support_resistance['support'])
            closest_resistance = self._find_closest_level(current_price, support_resistance['resistance'])

            # Calcular distancias relativas
            support_distance = abs(current_price - closest_support) / current_price if closest_support else float('inf')
            resistance_distance = abs(current_price - closest_resistance) / current_price if closest_resistance else float('inf')

            # Determinar posición en estructura
            structure_position = self._determine_price_position(
                current_price,
                market_structure['structure']['key_levels']
            )

            return {
                'price': current_price,
                'closest_levels': {
                    'support': closest_support,
                    'support_distance': support_distance,
                    'resistance': closest_resistance,
                    'resistance_distance': resistance_distance
                },
                'structure_position': structure_position,
                'is_at_level': min(support_distance, resistance_distance) <= self.price_params["level_proximity"],
                'position_strength': self._calculate_position_strength(
                    current_price,
                    closest_support,
                    closest_resistance
                )
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando precio actual: {str(e)}"))
            return self._get_default_price_analysis()

    def _find_pivot_points(self, prices: List[float], pivot_type: str) -> List[float]:
        """
        Encuentra puntos pivot (máximos o mínimos)
        """
        pivots = []
        window = self.price_params["pivot_periods"]

        for i in range(window, len(prices) - window):
            if pivot_type == 'high':
                if prices[i] == max(prices[i-window:i+window+1]):
                    pivots.append(prices[i])
            else:  # pivot_type == 'low'
                if prices[i] == min(prices[i-window:i+window+1]):
                    pivots.append(prices[i])

        return pivots

    def _cluster_levels(self, levels: List[float]) -> List[float]:
        """
        Agrupa niveles cercanos
        """
        if not levels:
            return []

        clustered = []
        current_cluster = [levels[0]]

        for level in levels[1:]:
            if abs(level - np.mean(current_cluster)) / np.mean(current_cluster) <= self.price_params["zone_thickness"]:
                current_cluster.append(level)
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]

        if current_cluster:
            clustered.append(np.mean(current_cluster))

        return clustered

    def _validate_levels(self, levels: List[float], prices: List[float], level_type: str) -> List[Dict]:
        """
        Valida la fuerza de los niveles
        """
        validated = []
        for level in levels:
            touches = self._count_touches(level, prices, level_type)
            if touches >= self.price_params["level_strength"]:
                validated.append({
                    'price': level,
                    'touches': touches,
                    'strength': min(1.0, touches / self.price_params["level_strength"]),
                    'type': level_type
                })

        return validated

    def _count_touches(self, level: float, prices: List[float], level_type: str) -> int:
        """
        Cuenta el número de veces que el precio toca un nivel
        """
        touches = 0
        threshold = level * self.price_params["zone_thickness"]

        for i in range(len(prices)):
            if level_type == 'resistance' and abs(prices[i] - level) <= threshold:
                touches += 1
            elif level_type == 'support' and abs(prices[i] - level) <= threshold:
                touches += 1

        return touches

    def _calculate_dynamic_levels(self, closes: List[float]) -> List[Dict]:
        """
        Calcula niveles dinámicos (EMAs, etc.)
        """
        try:
            ema20 = self._calculate_ema(closes, 20)
            ema50 = self._calculate_ema(closes, 50)
            ema200 = self._calculate_ema(closes, 200)

            return [
                {'price': ema20[-1], 'type': 'ema20', 'strength': 0.6},
                {'price': ema50[-1], 'type': 'ema50', 'strength': 0.7},
                {'price': ema200[-1], 'type': 'ema200', 'strength': 0.9}
            ]

        except Exception:
            return []

    def _calculate_ema(self, values: List[float], period: int) -> List[float]:
        """
        Calcula EMA (Exponential Moving Average)
        """
        if not values or period <= 0:
            return []

        ema = [values[0]]
        multiplier = 2 / (period + 1)

        for value in values[1:]:
            ema.append(value * multiplier + ema[-1] * (1 - multiplier))

        return ema

    def _select_most_relevant_levels(self, levels: List[Dict], max_levels: int = 3) -> List[Dict]:
        """
        Selecciona los niveles más relevantes basado en su fuerza
        """
        return sorted(levels, key=lambda x: x['strength'], reverse=True)[:max_levels]

    def _find_closest_level(self, price: float, levels: List[Dict]) -> Optional[float]:
        """
        Encuentra el nivel más cercano al precio actual
        """
        if not levels:
            return None

        closest = min(levels, key=lambda x: abs(x['price'] - price))
        return closest['price']

    def _calculate_position_strength(self, price: float, support: float, resistance: float) -> float:
        """
        Calcula la fortaleza de la posición actual del precio
        """
        if not support or not resistance:
            return 0.0

        # Calcular distancia relativa a los niveles
        range_size = resistance - support
        if range_size <= 0:
            return 0.0

        position = (price - support) / range_size

        # Fortaleza máxima en el medio del rango
        return 1.0 - abs(0.5 - position) * 2

    def _validate_data(self, candlesticks: List[Dict]) -> bool:
        """
        Valida que haya suficientes datos para el análisis
        """
        return bool(candlesticks and len(candlesticks) >= self.price_params["pivot_periods"] * 2)

    def _get_default_analysis(self) -> Dict:
        """
        Retorna un análisis por defecto
        """
        return {
            'support_resistance': {
                'support': [],
                'resistance': [],
                'dynamic': [],
                'zones': []
            },
            'market_structure': self._get_default_structure(),
            'key_zones': [],
            'current_analysis': self._get_default_price_analysis(),
            'is_valid': False,
            'signals': []
        }

    def _get_default_structure(self) -> Dict:
        """
        Retorna una estructura por defecto
        """
        return {
            'structure': {
                'trend': 'neutral',
                'key_levels': []
            },
            'breakouts': [],
            'swings': {
                'highs': [],
                'lows': []
            },
            'trend': 'neutral'
        }

    def _get_default_price_analysis(self) -> Dict:
        """
        Retorna un análisis de precio por defecto
        """
        return {
            'price': 0,
            'closest_levels': {
                'support': None,
                'support_distance': float('inf'),
                'resistance': None,
                'resistance_distance': float('inf')
            },
            'structure_position': 'undefined',
            'is_at_level': False,
            'position_strength': 0.0
        }

    def _determine_price_position(self, price: float, key_levels: List[Dict]) -> str:
        """
        Determina la posición del precio en la estructura del mercado
        """
        try:
            if not key_levels:
                return 'undefined'

            # Ordenar niveles por precio
            sorted_levels = sorted(key_levels, key=lambda x: x['price'])

            # Encontrar posición relativa
            for i, level in enumerate(sorted_levels):
                if price < level['price']:
                    if i == 0:
                        return 'below_structure'
                    else:
                        prev_level = sorted_levels[i-1]
                        return f"between_{prev_level['type']}_{level['type']}"

            return 'above_structure'

        except Exception as e:
            print(ConsoleColors.error(f"Error determinando posición del precio: {str(e)}"))
            return 'undefined'

    def _generate_price_signals(self, support_resistance: Dict,
                              market_structure: Dict,
                              current_analysis: Dict) -> List[str]:
        """
        Genera señales basadas en el análisis de precios
        """
        signals = []

        try:
            # Señales de niveles cercanos
            if current_analysis['is_at_level']:
                if current_analysis['closest_levels']['support_distance'] <= self.price_params["level_proximity"]:
                    signals.append(f"Precio en soporte (${current_analysis['closest_levels']['support']:.2f})")
                if current_analysis['closest_levels']['resistance_distance'] <= self.price_params["level_proximity"]:
                    signals.append(f"Precio en resistencia (${current_analysis['closest_levels']['resistance']:.2f})")

            # Señales de estructura
            if market_structure['trend'] != 'neutral':
                signals.append(f"Estructura de mercado {market_structure['trend']}")

            # Señales de zonas
            for zone in support_resistance['zones']:
                if zone['strength'] >= 2:  # Zonas con al menos 2 confluencias
                    zone_type = '/'.join(set(zone['types']))
                    signals.append(f"Zona fuerte de {zone_type} detectada (${zone['start']:.2f}-${zone['end']:.2f})")

            # Señales de ruptura
            if market_structure.get('breakouts'):
                for breakout in market_structure['breakouts']:
                    signals.append(f"Ruptura de {breakout['type']} en ${breakout['price']:.2f}")

            return signals

        except Exception as e:
            print(ConsoleColors.error(f"Error generando señales de precio: {str(e)}"))
            return []

    def _identify_breakout_levels(self, closes: List[float], trend: str, key_levels: List[Dict]) -> List[Dict]:
        """
        Identifica niveles de ruptura en la estructura
        """
        try:
            breakouts = []
            current_price = closes[-1]

            for level in key_levels:
                # Ruptura alcista
                if trend == 'bullish' and current_price > level['price'] * (1 + self.price_params["structure_params"]["structure_break"]):
                    breakouts.append({
                        'price': level['price'],
                        'type': 'resistance',
                        'strength': level.get('strength', 0.5),
                        'confirmed': self._confirm_breakout(closes, level['price'], 'bullish')
                    })
                # Ruptura bajista
                elif trend == 'bearish' and current_price < level['price'] * (1 - self.price_params["structure_params"]["structure_break"]):
                    breakouts.append({
                        'price': level['price'],
                        'type': 'support',
                        'strength': level.get('strength', 0.5),
                        'confirmed': self._confirm_breakout(closes, level['price'], 'bearish')
                    })

            return breakouts

        except Exception as e:
            print(ConsoleColors.error(f"Error identificando niveles de ruptura: {str(e)}"))
            return []

    def _confirm_breakout(self, closes: List[float], level: float, direction: str) -> bool:
        """
        Confirma si una ruptura es válida
        """
        try:
            # Usar últimas n velas para confirmar
            confirmation_period = 3
            recent_closes = closes[-confirmation_period:]

            if direction == 'bullish':
                return all(close > level for close in recent_closes)
            else:  # bearish
                return all(close < level for close in recent_closes)

        except Exception as e:
            print(ConsoleColors.error(f"Error confirmando ruptura: {str(e)}"))
            return False

    def _determine_structure_trend(self, structure: Dict) -> str:
        """
        Determina la tendencia basada en la estructura del mercado
        """
        try:
            if not structure or 'swings' not in structure:
                return 'neutral'

            # Analizar últimos swings
            recent_highs = structure['swings']['highs'][-3:]
            recent_lows = structure['swings']['lows'][-3:]

            if not recent_highs or not recent_lows:
                return 'neutral'

            # Tendencia alcista: máximos y mínimos crecientes
            if all(recent_highs[i] > recent_highs[i-1] for i in range(1, len(recent_highs))) and \
               all(recent_lows[i] > recent_lows[i-1] for i in range(1, len(recent_lows))):
                return 'bullish'

            # Tendencia bajista: máximos y mínimos decrecientes
            if all(recent_highs[i] < recent_highs[i-1] for i in range(1, len(recent_highs))) and \
               all(recent_lows[i] < recent_lows[i-1] for i in range(1, len(recent_lows))):
                return 'bearish'

            return 'neutral'

        except Exception as e:
            print(ConsoleColors.error(f"Error determinando tendencia de estructura: {str(e)}"))
            return 'neutral'

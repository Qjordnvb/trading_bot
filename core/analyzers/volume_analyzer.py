from typing import Dict, List, Optional
import numpy as np
from utils.console_colors import ConsoleColors

class VolumeAnalyzer:
    def __init__(self):
        self.volume_thresholds = {
            "significant": 2.0,    # 2x sobre el promedio
            "moderate": 1.5,       # 1.5x sobre el promedio
            "low": 0.5,           # 0.5x bajo el promedio
            "volume_spike": 3.0    # 3x sobre el promedio para considerar spike
        }

        self.analysis_periods = {
            "short_term": 5,     # Análisis de corto plazo
            "medium_term": 20,   # Análisis de medio plazo
            "long_term": 50      # Análisis de largo plazo
        }

        self.min_requirements = {
            "min_volume_24h": 1000000,     # Volumen mínimo en 24h
            "min_trades": 1000,            # Mínimo número de trades
            "min_liquidity_ratio": 0.02    # Ratio mínimo de liquidez
        }

    def analyze(self, candlesticks: List[Dict]) -> Dict:
        """
        Analiza el patrón de volumen y liquidez
        """
        try:
            if not self._validate_data(candlesticks):
                return self._get_default_analysis()

            # Extraer volúmenes y precios
            volumes, closes = self._extract_volume_data(candlesticks)
            if not volumes or not closes:
                return self._get_default_analysis()

            # Análisis básico de volumen
            volume_metrics = self._calculate_volume_metrics(volumes)

            # Análisis de presión compradora/vendedora
            pressure_analysis = self._analyze_buying_pressure(volumes, closes)

            # Análisis de tendencia de volumen
            trend_analysis = self._analyze_volume_trend(volumes)

            # Análisis de liquidez
            liquidity_analysis = self._analyze_liquidity(volumes, closes)

            return {
                'ratio': volume_metrics['volume_ratio'],
                'is_significant': volume_metrics['is_significant'],
                'is_increasing': trend_analysis['is_increasing'],
                'average': volume_metrics['average_volume'],
                'buy_pressure': pressure_analysis['buy_pressure'],
                'volume_trend': trend_analysis['trend'],
                'spikes': trend_analysis['spikes'],
                'liquidity': liquidity_analysis,
                'metrics': volume_metrics,
                'signals': self._generate_volume_signals(
                    volume_metrics,
                    pressure_analysis,
                    trend_analysis,
                    liquidity_analysis
                )
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error en análisis de volumen: {str(e)}"))
            return self._get_default_analysis()

    def _calculate_volume_metrics(self, volumes: List[float]) -> Dict:
        """Calcula métricas básicas de volumen"""
        try:
            recent_volumes = volumes[-self.analysis_periods["medium_term"]:]
            avg_volume = sum(recent_volumes) / len(recent_volumes)
            current_volume = volumes[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

            return {
                'volume_ratio': volume_ratio,
                'is_significant': volume_ratio > self.volume_thresholds["significant"],
                'average_volume': avg_volume,
                'current_volume': current_volume,
                'volume_change': self._calculate_volume_change(volumes),
                'consistency': self._calculate_volume_consistency(volumes)
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando métricas de volumen: {str(e)}"))
            return {
                'volume_ratio': 1.0,
                'is_significant': False,
                'average_volume': 0,
                'current_volume': 0,
                'volume_change': 0,
                'consistency': 0
            }

    def _analyze_buying_pressure(self, volumes: List[float], closes: List[float]) -> Dict:
        """Analiza la presión compradora vs vendedora"""
        try:
            recent_volumes = volumes[-self.analysis_periods["short_term"]:]
            recent_closes = closes[-self.analysis_periods["short_term"]:]

            if len(recent_volumes) != len(recent_closes):
                return {'buy_pressure': 0.5, 'momentum': 0}

            buy_volume = sum(
                vol for vol, close_change in zip(
                    recent_volumes,
                    [closes[i] - closes[i-1] for i in range(-self.analysis_periods["short_term"], 0)]
                ) if close_change > 0
            )

            total_volume = sum(recent_volumes)
            buy_pressure = buy_volume / total_volume if total_volume > 0 else 0.5

            return {
                'buy_pressure': buy_pressure,
                'momentum': self._calculate_volume_momentum(volumes),
                'pressure_trend': 'increasing' if buy_pressure > 0.6 else 'decreasing' if buy_pressure < 0.4 else 'neutral'
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando presión compradora: {str(e)}"))
            return {'buy_pressure': 0.5, 'momentum': 0, 'pressure_trend': 'neutral'}

    def _analyze_volume_trend(self, volumes: List[float]) -> Dict:
        """Analiza la tendencia del volumen"""
        try:
            # Medias móviles de volumen
            short_ma = sum(volumes[-5:]) / 5
            long_ma = sum(volumes[-20:]) / 20

            # Detectar spikes de volumen
            recent_std = np.std(volumes[-20:])
            volume_spikes = [
                i for i, vol in enumerate(volumes[-5:])
                if vol > (sum(volumes[-25:-5]) / 20 + 2 * recent_std)
            ]

            return {
                'trend': 'increasing' if short_ma > long_ma else 'decreasing',
                'strength': abs((short_ma - long_ma) / long_ma) if long_ma > 0 else 0,
                'is_increasing': short_ma > long_ma,
                'spikes': {
                    'count': len(volume_spikes),
                    'positions': volume_spikes,
                    'significance': 'high' if len(volume_spikes) > 2 else 'moderate' if volume_spikes else 'low'
                }
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando tendencia de volumen: {str(e)}"))
            return {
                'trend': 'neutral',
                'strength': 0,
                'is_increasing': False,
                'spikes': {'count': 0, 'positions': [], 'significance': 'low'}
            }

    def _analyze_liquidity(self, volumes: List[float], closes: List[float]) -> Dict:
        """Analiza la liquidez del mercado"""
        try:
            avg_volume = sum(volumes) / len(volumes)
            volume_stability = 1 - (np.std(volumes) / avg_volume)

            # Calcular profundidad de mercado estimada
            price_impact = self._estimate_price_impact(volumes, closes)

            return {
                'is_liquid': avg_volume >= self.min_requirements["min_volume_24h"],
                'stability': volume_stability,
                'depth': {
                    'price_impact': price_impact,
                    'rating': 'high' if price_impact < 0.01 else 'moderate' if price_impact < 0.03 else 'low'
                },
                'quality': self._calculate_liquidity_quality(volumes, volume_stability)
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando liquidez: {str(e)}"))
            return {
                'is_liquid': False,
                'stability': 0,
                'depth': {'price_impact': 1, 'rating': 'low'},
                'quality': 'poor'
            }

    def _calculate_volume_change(self, volumes: List[float]) -> float:
        """Calcula el cambio porcentual en el volumen"""
        try:
            if len(volumes) < 2:
                return 0.0
            return ((volumes[-1] - volumes[-2]) / volumes[-2]) * 100
        except Exception:
            return 0.0

    def _calculate_volume_consistency(self, volumes: List[float]) -> float:
        """Calcula la consistencia del volumen"""
        try:
            if len(volumes) < 20:
                return 0.0
            std_dev = np.std(volumes[-20:])
            mean_volume = np.mean(volumes[-20:])
            return 1 - (std_dev / mean_volume) if mean_volume > 0 else 0.0
        except Exception:
            return 0.0

    def _calculate_volume_momentum(self, volumes: List[float]) -> float:
        """Calcula el momentum del volumen"""
        try:
            if len(volumes) < 10:
                return 0.0
            recent_avg = sum(volumes[-5:]) / 5
            previous_avg = sum(volumes[-10:-5]) / 5
            return ((recent_avg - previous_avg) / previous_avg) * 100 if previous_avg > 0 else 0.0
        except Exception:
            return 0.0

    def _estimate_price_impact(self, volumes: List[float], closes: List[float]) -> float:
        """Estima el impacto en el precio por unidad de volumen"""
        try:
            if len(volumes) < 2 or len(closes) < 2:
                return 1.0
            price_changes = [abs(closes[i] - closes[i-1]) for i in range(1, len(closes))]
            return np.mean([change/volume for change, volume in zip(price_changes, volumes[1:])]) if volumes[1:] else 1.0
        except Exception:
            return 1.0

    def _calculate_liquidity_quality(self, volumes: List[float], stability: float) -> str:
        """Evalúa la calidad de la liquidez"""
        try:
            avg_volume = sum(volumes) / len(volumes)
            if avg_volume >= self.min_requirements["min_volume_24h"] * 2 and stability >= 0.7:
                return 'excellent'
            elif avg_volume >= self.min_requirements["min_volume_24h"] and stability >= 0.5:
                return 'good'
            elif avg_volume >= self.min_requirements["min_volume_24h"] * 0.5:
                return 'moderate'
            return 'poor'
        except Exception:
            return 'poor'

    def _generate_volume_signals(self, metrics: Dict, pressure: Dict,
                               trend: Dict, liquidity: Dict) -> List[str]:
        """Genera señales basadas en el análisis de volumen"""
        signals = []

        # Señales de volumen significativo
        if metrics['is_significant']:
            signals.append(f"Volumen significativo ({metrics['volume_ratio']:.1f}x promedio)")

        # Señales de presión compradora
        if pressure['buy_pressure'] > 0.6:
            signals.append(f"Alta presión compradora ({pressure['buy_pressure']:.1%})")
        elif pressure['buy_pressure'] < 0.4:
            signals.append(f"Alta presión vendedora ({1-pressure['buy_pressure']:.1%})")

        # Señales de tendencia
        if trend['is_increasing'] and trend['strength'] > 0.2:
            signals.append(f"Tendencia de volumen alcista ({trend['strength']:.1%})")

        # Señales de liquidez
        if not liquidity['is_liquid']:
            signals.append("Baja liquidez - Precaución")
        elif liquidity['quality'] == 'excellent':
            signals.append("Excelente liquidez y profundidad de mercado")

        return signals

    def _extract_volume_data(self, candlesticks: List[Dict]) -> tuple:
        """Extrae volúmenes y precios de cierre de las velas"""
        volumes = []
        closes = []

        for candle in candlesticks:
            try:
                if isinstance(candle, dict):
                    volumes.append(float(candle['volume']))
                    closes.append(float(candle['close']))
                elif isinstance(candle, (list, tuple)):
                    volumes.append(float(candle[5]))  # Volumen en índice 5
                    closes.append(float(candle[4]))   # Cierre en índice 4
            except (IndexError, KeyError, ValueError):
                continue

        return volumes, closes

    def _validate_data(self, candlesticks: List[Dict]) -> bool:
        """Valida que haya suficientes datos para el análisis"""
        return bool(candlesticks and len(candlesticks) >= self.analysis_periods["long_term"])

    def _get_default_analysis(self) -> Dict:
        """Retorna un análisis por defecto cuando hay error o datos insuficientes"""
        return {
            'ratio': 1.0,
            'is_significant': False,
            'is_increasing': False,
            'average': 0,
            'buy_pressure': 0.5,
            'volume_trend': 'neutral',
            'spikes': {'count': 0, 'positions': [], 'significance': 'low'},
            'liquidity': {'is_liquid': False, 'quality': 'poor'},
            'metrics': {},
            'signals': ["Datos insuficientes para análisis"]
        }

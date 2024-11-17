# core/market_analyzer.py
import numpy as np
from typing import Dict, List, Optional, Tuple
from utils.console_colors import ConsoleColors
from alerts.alert_manager import AlertManager
from models.data_classes import TradeRecommendation, TimingWindow
from models.enums import MarketTrend, TradingSignal, SignalStrength, EntryTiming

class MarketAnalyzer:
    def __init__(self, client, alert_manager=None):
        self.client = client
        self.alert_manager = alert_manager

        # Parámetros optimizados de la estrategia
        self.strategy_params = {
            # EMAs (Medias Móviles Exponenciales)
            "ema_periods": {
                "fast": 8,      # Señales rápidas de entrada/salida
                "medium": 21,   # Confirmación de tendencia a medio plazo
                "slow": 50,     # Confirmación de tendencia a largo plazo
                "trend": 200,   # Tendencia principal del mercado
                "validation_periods": [13, 34, 89],  # Fibonacci EMAs para confirmación
                "min_separation": 0.5,  # % mínimo de separación entre EMAs
                "alignment_threshold": 0.8  # % de EMAs que deben estar alineadas
            },


            # MACD (Moving Average Convergence Divergence)
            "macd_params": {
                "fast": 12,     # EMA rápida para MACD
                "slow": 26,     # EMA lenta para MACD
                "signal": 9,    # Línea de señal
                # Optimizaciones adicionales
                "min_hist_strength": 0.2,  # Fuerza mínima del histograma
                "confirmation_periods": 3,  # Períodos de confirmación
                "divergence_threshold": 0.1,  # % para confirmar divergencias
                "volume_confirmation": True  # Requerir confirmación de volumen
            },

            # Bandas de Bollinger
            "bollinger_params": {
                "period": 20,    # Período para cálculo
                "std_dev": 2,    # Desviaciones estándar
                # Optimizaciones adicionales
                "squeeze_threshold": 0.5,  # % para detectar compresión
                "expansion_threshold": 2.5,  # % para detectar expansión
                "touch_count": 2,  # Toques necesarios para validación
                "bandwidth_min": 1.0,  # Ancho mínimo de bandas
                "percent_b_threshold": 0.05  # % B para confirmación
            },

            # RSI (Índice de Fuerza Relativa)
            "rsi_params": {
                "period": 14,    # Período de cálculo
                "oversold": 30,  # Nivel de sobreventa
                "overbought": 70,  # Nivel de sobrecompra
                # Optimizaciones adicionales
                "divergence_periods": 5,  # Períodos para buscar divergencias
                "trend_alignment": True,  # Requerir alineación con tendencia
                "confirmation_levels": {  # Niveles adicionales de confirmación
                    "extreme_oversold": 20,
                    "extreme_overbought": 80,
                    "neutral_low": 45,
                    "neutral_high": 55
                },
                "momentum_threshold": 0.3  # % para cambio de momentum
            },

            # Gestión de Riesgo
            "risk_params": {
                "max_risk_percent": 2,    # Riesgo máximo por operación
                "partial_tp_ratio": 1.5,  # Ratio para toma parcial de beneficios
                "max_trades_per_day": 3,  # Máximo de operaciones diarias
                "volatility_threshold": 3.0,  # Desviación estándar máxima
                "position_sizing":{
                    "base_size": 1.0,  # Tamaño base de posición
                    "scale_in_levels": [0.3, 0.3, 0.4],  # Distribución de entradas
                    "max_position_size": 4.0  # Tamaño máximo total
                },
                "take_profit_levels": [  # Múltiples niveles de take profit
                    {"size": 0.3, "ratio": 1.5},  # 30% a ratio 1.5
                    {"size": 0.4, "ratio": 2.0},  # 40% a ratio 2.0
                    {"size": 0.3, "ratio": 3.0}   # 30% a ratio 3.0
                ],
                "stop_loss_adjustment": {
                    "breakeven_move": 0.5,  # Mover a breakeven al 50% del TP1
                    "trailing_activation": 1.0,  # Activar trailing al 100% del TP1
                    "trailing_step": 0.25  # Paso del trailing stop en ATR
                },
                "risk_adjustments": {
                    "trend_strength": 0.2,  # ±20% basado en fuerza de tendencia
                    "volatility": 0.3,     # ±30% basado en volatilidad
                    "market_condition": 0.25  # ±25% basado en condición de mercado
                },
                "filters": {
                    "min_volume_24h": 1000000,  # Volumen mínimo en 24h
                    "min_liquidity_ratio": 0.02,  # Ratio mínimo de liquidez
                    "max_spread_percent": 0.1,  # Spread máximo permitido
                    "news_impact_delay": 30  # Minutos de espera post-noticias
                },
                "risk": {
                "max_loss_percent": 5,      # Máxima pérdida permitida
                "min_risk_reward": 2,       # Mínimo ratio riesgo/beneficio
                "max_position_size": 4.0,   # Tamaño máximo de posición
                "volatility_adjustment": 1.5 # Factor de ajuste por volatilidad
                },
                "entry": {
                "support_buffer": 1.01,     # 1% sobre soporte
                "resistance_buffer": 0.99,  # 1% bajo resistencia
                "price_buffer": 0.995       # 0.5% bajo precio actual
                },
                "signals": {
                "trend_strength_threshold": 0.6,  # Mínima fuerza de tendencia
                "volume_significance": 1.5,       # Multiplicador de volumen
                "rsi_oversold": 30,              # Nivel de sobreventa
                "rsi_overbought": 70,            # Nivel de sobrecompra
                "price_level_threshold": 0.02     # 2% para niveles de precio
                }
            },

            # Análisis Técnico Avanzado
            "advanced_analysis": {
                "pattern_recognition": {
                    "min_pattern_size": 0.01,  # Tamaño mínimo del patrón
                    "confirmation_candles": 2,  # Velas de confirmación
                    "volume_confirmation": True  # Requerir confirmación de volumen
                },
                "support_resistance": {
                    "pivot_periods": 20,  # Períodos para pivotes
                    "level_strength": 3,  # Toques para confirmar nivel
                    "zone_thickness": 0.002  # Grosor de zonas S/R
                },
                "market_structure": {
                    "swing_threshold": 0.01,  # % para identificar swings
                    "trend_validation": 3,  # Pivotes para confirmar tendencia
                    "structure_break": 0.005  # % para ruptura de estructura
                }
            }
        },

        self.thresholds = {
            "momentum": {
                "strong_buy": 15,
                "buy": 5,
                "strong_sell": -15,
                "sell": -5
            },
            "volume": {
                "significant": 2,
                "moderate": 1.5,
                "low": 0.5
            },
            "rsi": {
                "strong_oversold": 20,
                "oversold": 30,
                "overbought": 70,
                "strong_overbought": 80
            }
        },
        self.timing_thresholds = {
            "oversold_rsi": 30,
            "overbought_rsi": 70,
            "price_support": 0.05,
            "price_resistance": 0.05,
            "volume_spike": 2.0,
            "volume_significant": 1.5,
            "consolidation_range": 0.02,
            "high_volatility": 0.05,
            "low_volatility": 0.01
        },
        self.signal_thresholds = {
            'score': {
                'strong_buy': 80,    # Puntuación mínima para compra fuerte
                'buy': 65,           # Puntuación mínima para compra
                'strong_sell': -80,  # Puntuación máxima para venta fuerte
                'sell': -65          # Puntuación máxima para venta
            },
            'confirmations': {
                'strong': 6,         # Confirmaciones necesarias para señal fuerte
                'moderate': 4,       # Confirmaciones para señal moderada
                'weak': 2            # Confirmaciones para señal débil
            }
        }

    def _validate_market_conditions(self, data: Dict) -> bool:
        """Validación avanzada de condiciones de mercado"""
        try:
            # 1. Verificar volatilidad
            volatility = self._calculate_volatility(data)
            if volatility > self.strategy_params["risk_params"]["volatility_threshold"]:
                return False

            # 2. Verificar volumen
            volume_24h = float(data.get('volume', 0))
            if volume_24h < self.strategy_params["risk_params"]["filters"]["min_volume_24h"]:
                return False

            # 3. Verificar spread
            current_spread = self._calculate_spread(data)
            if current_spread > self.strategy_params["risk_params"]["filters"]["max_spread_percent"]:
                return False

            # 4. Verificar liquidez
            liquidity_ratio = self._calculate_liquidity_ratio(data)
            if liquidity_ratio < self.strategy_params["risk_params"]["filters"]["min_liquidity_ratio"]:
                return False

            return True

        except Exception as e:
            print(ConsoleColors.error(f"Error validando condiciones de mercado: {str(e)}"))
            return False

    def _analyze_market_conditions(self, candlesticks: List[Dict]) -> Dict:
        """Analiza las condiciones generales del mercado"""
        try:
            # Calcular volatilidad
            volatility = self._calculate_volatility(candlesticks)

            # Analizar volumen
            volume_analysis = self._analyze_volume(candlesticks)

            # Calcular momentum
            momentum = self._analyze_momentum(candlesticks)

            # Verificar condiciones para operar
            is_tradeable = (
                volatility <= self.strategy_params["risk_params"]["volatility_threshold"] and
                volume_analysis['ratio'] >= self.thresholds['volume']['significant'] and
                self._validate_volume(candlesticks)
            )

            return {
                'is_tradeable': is_tradeable,
                'reason': self._generate_market_condition_reason(
                    volatility,
                    volume_analysis,
                    momentum
                ),
                'data': candlesticks,
                'volatility': volatility,
                'volume': volume_analysis,
                'momentum': momentum
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando condiciones de mercado: {str(e)}"))
            return {
                'is_tradeable': False,
                'reason': "Error en análisis de condiciones de mercado",
                'data': candlesticks
            }

    def _analyze_technical_indicators(self, candlesticks):
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

            # Extraer datos de las velas
            closes = []
            highs = []
            lows = []

            for candle in candlesticks:
                try:
                    if isinstance(candle, (list, tuple)):
                        closes.append(float(candle[4]))  # close
                        highs.append(float(candle[2]))   # high
                        lows.append(float(candle[3]))    # low
                    elif isinstance(candle, dict):
                        closes.append(float(candle['close']))
                        highs.append(float(candle['high']))
                        lows.append(float(candle['low']))
                except (IndexError, KeyError, ValueError, TypeError) as e:
                    print(f"Error procesando vela: {e}")
                    continue

            if len(closes) < 50:  # Necesitamos suficientes datos
                return {
                    'rsi': 50,
                    'macd': {'trend': 'neutral', 'histogram': 0, 'crossover': 'none'},
                    'bollinger': {'upper': 0, 'middle': 0, 'lower': 0},
                    'is_valid': False
                }

            # Calcular indicadores
            rsi = self._calculate_rsi(closes[-14:])
            macd = self._calculate_macd(closes)
            bb = self._calculate_bollinger_bands(closes[-20:])

            return {
                'rsi': rsi,
                'macd': macd,
                'bollinger': bb,
                'is_valid': True,
                'last_price': closes[-1]
            }

        except Exception as e:
            print(f"Error en análisis técnico: {str(e)}")
            print(f"Tipo de candlesticks: {type(candlesticks)}")
            if candlesticks and len(candlesticks) > 0:
                print(f"Primera vela: {candlesticks[0]}")
            return {
                'rsi': 50,
                'macd': {'trend': 'neutral', 'histogram': 0, 'crossover': 'none'},
                'bollinger': {'upper': 0, 'middle': 0, 'lower': 0},
                'is_valid': False
            }

    def _calculate_rsi(self, prices, period: int = 14) -> float:
        """
        Calcula el RSI (Relative Strength Index)
        """
        try:
            # Si prices es una lista de diccionarios, extraer solo los precios de cierre
            if isinstance(prices[0], dict):
                prices = [float(price['close']) for price in prices]
            # Si prices es una lista de listas/tuplas (formato Binance)
            elif isinstance(prices[0], (list, tuple)):
                prices = [float(price[4]) for price in prices]  # índice 4 es el precio de cierre

            # Convertir a numpy array
            prices = np.array(prices, dtype=float)

            if len(prices) < period + 1:
                return 50.0

            # Calcular cambios
            deltas = np.diff(prices)

            # Separar ganancias y pérdidas
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            # Calcular promedios
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])

            if avg_loss == 0:
                return 100.0

            if avg_gain == 0:
                return 0.0

            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

            return min(100.0, max(0.0, float(rsi)))  # Asegurar que esté entre 0 y 100

        except Exception as e:
            print(f"Error calculando RSI: {str(e)}")
            if len(prices) > 0:
                print(f"Primer elemento de prices: {prices[0]}")
                print(f"Tipo de prices: {type(prices)}")
            return 50.0  # Valor neutral en caso de error



    def analyze_trading_opportunity(self, symbol: str) -> Optional[TradeRecommendation]:
        """
        Analiza una oportunidad de trading con niveles de precio optimizados
        """
        try:
            # Obtener datos actualizados
            market_data = {
                'D1': self.client.get_klines(symbol, interval='1d', limit=200),
                'H4': self.client.get_klines(symbol, interval='4h', limit=200),
                'H1': self.client.get_klines(symbol, interval='1h', limit=200)
            }

            if not all(market_data.values()):
                return self._generate_hold_recommendation("Datos de mercado insuficientes")

            try:
                current_price = float(self.client.get_ticker_price(symbol)['price'])
            except Exception as e:
                print(f"Error obteniendo precio actual: {str(e)}")
                return self._generate_hold_recommendation("Error obteniendo precio actual")

            # Análisis técnico
            trend_analysis = self._analyze_main_trend(market_data)
            technical_analysis = self._analyze_technical_indicators(market_data['H4'])
            volume_analysis = self._analyze_volume(market_data['H4'])
            support_resistance = self._calculate_support_resistance(market_data['H4'])
            volatility = self._calculate_volatility(market_data['H4'])

            # Determinar señal y fuerza
            signal, strength = self._determine_trading_signal(
                trend_analysis,
                technical_analysis,
                volume_analysis,
                support_resistance,
                current_price,
                volatility
            )

            # Generar razones del análisis
            reasons = self._generate_analysis_reasons(
                trend_analysis,
                technical_analysis,
                volume_analysis,
                support_resistance,
                volatility
            )

            # Calcular niveles de precio optimizados
            price_levels = self._calculate_optimal_price_levels(
                current_price,
                support_resistance,
                volatility,
                trend_analysis,
                technical_analysis
            )

            return TradeRecommendation(
                signal=signal,
                strength=strength,
                reasons=reasons,
                entry_price=price_levels['entry'],
                stop_loss=price_levels['stop_loss'],
                take_profit=price_levels['take_profit']
            )

        except Exception as e:
            print(f"Error analizando {symbol}: {str(e)}")
            return self._generate_hold_recommendation(f"Error en análisis: {str(e)}")

    def _determine_trading_signal(self, trend_analysis, technical_analysis,
                                volume_analysis, support_resistance, current_price,
                                volatility) -> Tuple[TradingSignal, SignalStrength]:
        """
        Sistema mejorado de señales de trading con una precisión objetivo del 95%
        utilizando un sistema de scoring ponderado y múltiples confirmaciones.
        """
        try:
            # 1. Inicializar sistema de puntuación
            score = 0.0
            confirmations = 0
            conditions_met = []

            # 2. Análisis de Tendencia (40 puntos máximo)
            if trend_analysis:
                trend = trend_analysis.get('trend')
                strength = trend_analysis.get('strength', 0)

                if trend == 'bullish':
                    trend_score = 40 * strength
                    score += trend_score
                    if strength > 0.6:
                        confirmations += 2
                    elif strength > 0.3:
                        confirmations += 1
                    conditions_met.append(f'Tendencia alcista ({strength:.1%})')
                elif trend == 'bearish':
                    trend_score = -40 * strength
                    score += trend_score
                    if strength > 0.6:
                        confirmations += 2
                    elif strength > 0.3:
                        confirmations += 1
                    conditions_met.append(f'Tendencia bajista ({strength:.1%})')

            # 3. Análisis de Soporte/Resistencia (25 puntos máximo)
            if support_resistance:
                support = support_resistance.get('support', 0)
                resistance = support_resistance.get('resistance', 0)

                if support and resistance:
                    distance_to_support = (current_price - support) / support
                    distance_to_resistance = (resistance - current_price) / current_price

                    # Cerca del soporte (señal de compra)
                    if distance_to_support <= 0.02:  # 2% del soporte
                        score += 25
                        confirmations += 1
                        conditions_met.append('Precio cerca del soporte')
                    # Cerca de la resistencia (señal de venta)
                    elif distance_to_resistance <= 0.02:  # 2% de la resistencia
                        score -= 25
                        confirmations += 1
                        conditions_met.append('Precio cerca de la resistencia')

            # 4. Análisis de Volumen (20 puntos máximo)
            if volume_analysis:
                volume_ratio = volume_analysis.get('ratio', 1.0)
                is_increasing = volume_analysis.get('is_increasing', False)

                if volume_ratio > 2.0:  # Volumen significativo
                    if is_increasing:
                        score += 20
                        confirmations += 1
                        conditions_met.append('Volumen creciente significativo')
                    else:
                        score += 10
                elif volume_ratio < 0.5:  # Volumen bajo
                    score -= 10
                    conditions_met.append('Volumen bajo')

            # 5. Análisis Técnico (15 puntos máximo)
            if technical_analysis:
                # RSI
                rsi = technical_analysis.get('rsi', 50)
                if rsi <= 30:  # Sobreventa
                    score += 15
                    confirmations += 1
                    conditions_met.append(f'RSI en sobreventa ({rsi:.1f})')
                elif rsi >= 70:  # Sobrecompra
                    score -= 15
                    confirmations += 1
                    conditions_met.append(f'RSI en sobrecompra ({rsi:.1f})')

                # MACD
                macd = technical_analysis.get('macd', {})
                if macd.get('crossover') == 'bullish':
                    score += 10
                    confirmations += 1
                    conditions_met.append('Cruce alcista de MACD')
                elif macd.get('crossover') == 'bearish':
                    score -= 10
                    confirmations += 1
                    conditions_met.append('Cruce bajista de MACD')

            # Determinar señal y fuerza basado en score y confirmaciones
            if score >= self.signal_thresholds['score']['strong_buy'] and confirmations >= self.signal_thresholds['confirmations']['strong']:
                return TradingSignal.BUY, SignalStrength.STRONG
            elif score >= self.signal_thresholds['score']['buy'] and confirmations >= self.signal_thresholds['confirmations']['moderate']:
                return TradingSignal.BUY, SignalStrength.MODERATE
            elif score <= self.signal_thresholds['score']['strong_sell'] and confirmations >= self.signal_thresholds['confirmations']['strong']:
                return TradingSignal.SELL, SignalStrength.STRONG
            elif score <= self.signal_thresholds['score']['sell'] and confirmations >= self.signal_thresholds['confirmations']['moderate']:
                return TradingSignal.SELL, SignalStrength.MODERATE
            else:
                strength = SignalStrength.MODERATE if abs(score) > 30 else SignalStrength.WEAK
                return TradingSignal.HOLD, strength

        except Exception as e:
            print(ConsoleColors.error(f"Error en determine_trading_signal: {str(e)}"))
            return TradingSignal.HOLD, SignalStrength.WEAK

    def _calculate_optimal_price_levels(self, current_price, support_resistance,
                                      volatility, trend_analysis, technical_analysis) -> Dict:
        """
        Calcula niveles de precio óptimos basados en análisis técnico
        """
        try:
            # Valores iniciales basados en el precio actual
            price_levels = {
                'entry': current_price,
                'stop_loss': current_price * 0.95,  # -5% por defecto
                'take_profit': current_price * 1.15  # +15% por defecto
            }

            if support_resistance:
                support = support_resistance.get('support')
                resistance = support_resistance.get('resistance')

                if support and resistance:
                    # Entrada basada en niveles técnicos
                    if current_price < support * 1.02:  # Cerca del soporte
                        price_levels['entry'] = support * 1.01  # 1% sobre soporte
                        price_levels['stop_loss'] = support * 0.98  # 2% bajo soporte
                        price_levels['take_profit'] = min(
                            resistance * 0.99,  # Justo bajo resistencia
                            support * 1.10  # Mínimo 10% de ganancia
                        )
                    else:
                        # Entrada conservadora a precio actual
                        risk = current_price - support
                        reward = resistance - current_price

                        if reward/risk >= 2:  # Risk/Reward mínimo 1:2
                            price_levels['entry'] = current_price * 0.995  # 0.5% bajo precio actual
                            price_levels['stop_loss'] = max(
                                support * 0.99,
                                price_levels['entry'] * 0.97  # Máximo 3% de pérdida
                            )
                            price_levels['take_profit'] = min(
                                resistance * 0.99,
                                price_levels['entry'] * 1.06  # Mínimo 6% de ganancia
                            )

            # Ajustar por volatilidad
            if volatility > 0.05:  # Alta volatilidad
                price_levels['stop_loss'] = price_levels['entry'] * 0.93  # Stop más amplio
                price_levels['take_profit'] = price_levels['entry'] * 1.20  # Target más ambicioso

            return price_levels

        except Exception as e:
            print(f"Error en calculate_optimal_price_levels: {str(e)}")
            return {
                'entry': current_price,
                'stop_loss': current_price * 0.95,
                'take_profit': current_price * 1.15
            }

    def _generate_analysis_reasons(self, trend_analysis, technical_analysis,
                                 volume_analysis, support_resistance, volatility) -> List[str]:
        """
        Genera razones detalladas del análisis
        """
        reasons = []

        # Añadir razón de tendencia
        if trend_analysis.get('trend') == 'bullish':
            reasons.append(f"Tendencia alcista con fuerza {trend_analysis.get('strength', 0):.2%}")
        elif trend_analysis.get('trend') == 'bearish':
            reasons.append(f"Tendencia bajista con fuerza {trend_analysis.get('strength', 0):.2%}")

        # Añadir niveles de soporte/resistencia
        if support_resistance:
            if 'support' in support_resistance:
                reasons.append(f"Soporte en ${support_resistance['support']:.2f}")
            if 'resistance' in support_resistance:
                reasons.append(f"Resistencia en ${support_resistance['resistance']:.2f}")

        # Añadir información de volatilidad
        reasons.append(f"{'Alta' if volatility > 0.05 else 'Baja'} volatilidad ({volatility:.1%})")

        # Añadir información de volumen
        if volume_analysis.get('is_significant'):
            reasons.append(f"Volumen significativo ({volume_analysis.get('ratio', 0):.1f}x promedio)")

        return reasons



    def _generate_hold_recommendation(self, reason: str, current_price: float = None) -> TradeRecommendation:
        """
        Genera una recomendación de HOLD
        """
        try:
            if current_price is None:
                return TradeRecommendation(
                    signal=TradingSignal.HOLD,
                    strength=SignalStrength.WEAK,
                    reasons=[reason],
                    entry_price=None,
                    stop_loss=None,
                    take_profit=None
                )

            stop_loss = current_price * 0.95
            take_profit = current_price * 1.15

            return TradeRecommendation(
                signal=TradingSignal.HOLD,
                strength=SignalStrength.WEAK,
                reasons=[reason],
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
        except Exception as e:
            print(f"Error generando recomendación HOLD: {str(e)}")
            return TradeRecommendation(
                signal=TradingSignal.HOLD,
                strength=SignalStrength.WEAK,
                reasons=["Error en análisis"],
                entry_price=None,
                stop_loss=None,
                take_profit=None
            )

    def _analyze_main_trend(self, market_data):
        """
        Analiza la tendencia principal con los datos proporcionados.
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
                if not data or len(data) < 50:  # Mínimo de velas necesarias
                    continue

                # Extraer valores de cierre numéricos
                closes = []
                try:
                    for candle in data:
                        if isinstance(candle, dict):
                            close = float(candle.get('close', 0))
                            if close > 0:
                                closes.append(close)
                        elif isinstance(candle, (list, tuple)) and len(candle) >= 4:
                            # Si los datos vienen como lista [timestamp, open, high, low, close, ...]
                            close = float(candle[4])
                            if close > 0:
                                closes.append(close)
                except (ValueError, TypeError, IndexError):
                    continue

                if len(closes) < 50:
                    continue

                # Calcular EMAs
                ema20 = self._calculate_ema(closes, 20)[-1]
                ema50 = self._calculate_ema(closes, 50)[-1]
                ema200 = self._calculate_ema(closes, 200)[-1]

                # Calcular momentum
                momentum = ((closes[-1] - closes[-20]) / closes[-20]) * 100

                # Determinar tendencia por timeframe
                timeframe_trend = {
                    'price': closes[-1],
                    'ema20': ema20,
                    'ema50': ema50,
                    'ema200': ema200,
                    'momentum': momentum,
                    'is_bullish': closes[-1] > ema20 > ema50 > ema200,
                    'is_bearish': closes[-1] < ema20 < ema50 < ema200
                }

                trend_analysis['timeframes'][timeframe] = timeframe_trend
                trend_analysis['data'][timeframe] = closes

            # Analizar tendencia global
            timeframe_count = len(trend_analysis['timeframes'])
            if timeframe_count > 0:
                bullish_count = sum(1 for tf in trend_analysis['timeframes'].values() if tf['is_bullish'])
                bearish_count = sum(1 for tf in trend_analysis['timeframes'].values() if tf['is_bearish'])

                trend_analysis['is_valid'] = True
                if bullish_count > timeframe_count / 2:
                    trend_analysis['trend'] = 'bullish'
                    trend_analysis['strength'] = bullish_count / timeframe_count
                elif bearish_count > timeframe_count / 2:
                    trend_analysis['trend'] = 'bearish'
                    trend_analysis['strength'] = bearish_count / timeframe_count
                else:
                    trend_analysis['trend'] = 'neutral'
                    trend_analysis['strength'] = 0.5

            return trend_analysis

        except Exception as e:
            print(f"Error en análisis de tendencia principal: {str(e)}")
            return {
                'is_valid': False,
                'trend': 'neutral',
                'strength': 0,
                'data': {},
                'timeframes': {}
            }

    # Mantener métodos existentes
    def _analyze_trend(self, candlesticks: List[Dict]) -> MarketTrend:
        """Analiza la tendencia del mercado"""
        try:
            closes = [float(candle['close']) for candle in candlesticks]

            # Calcular EMAs
            ema20 = self._calculate_ema(closes, 20)
            ema50 = self._calculate_ema(closes, 50)

            # Calcular momentum
            momentum = (closes[-1] - closes[-20]) / closes[-20] * 100

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

    def _analyze_volume(self, candlesticks):
        """
        Analiza el patrón de volumen
        """
        try:
            # Validar que candlesticks no sea None y tenga elementos
            if not candlesticks or len(candlesticks) < 20:
                return {
                    'ratio': 1.0,
                    'is_significant': False,
                    'is_increasing': False,
                    'average': 0,
                    'buy_pressure': 0
                }

            # Extraer volúmenes y cierres
            volumes = []
            closes = []

            for candle in candlesticks:
                try:
                    # Si es una lista/tupla de la API de Binance:
                    # [timestamp, open, high, low, close, volume, ...]
                    if isinstance(candle, (list, tuple)):
                        vol = float(candle[5])  # El volumen está en el índice 5
                        close = float(candle[4])  # El cierre está en el índice 4
                        volumes.append(vol)
                        closes.append(close)
                except (IndexError, ValueError, TypeError):
                    continue

            if len(volumes) < 20:
                return {
                    'ratio': 1.0,
                    'is_significant': False,
                    'is_increasing': False,
                    'average': 0,
                    'buy_pressure': 0
                }

            # Cálculos de volumen
            avg_volume = sum(volumes[-20:]) / 20
            current_volume = volumes[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

            # Calcular presión compradora
            recent_volumes = volumes[-5:]
            recent_price_changes = [
                closes[i] - closes[i-1]
                for i in range(len(closes)-5, len(closes))
            ]

            buy_volume = sum(
                vol for vol, change in zip(recent_volumes, recent_price_changes)
                if change > 0
            )
            total_volume = sum(recent_volumes)
            buy_pressure = buy_volume / total_volume if total_volume > 0 else 0.5

            return {
                'ratio': volume_ratio,
                'is_significant': volume_ratio > self.thresholds['volume']['significant'],
                'is_increasing': sum(volumes[-3:]) > sum(volumes[-6:-3]),
                'average': avg_volume,
                'buy_pressure': buy_pressure
            }

        except Exception as e:
            print(f"Error en análisis de volumen: {e}")
            print(f"Tipo de candlesticks: {type(candlesticks)}")
            if candlesticks and len(candlesticks) > 0:
                print(f"Tipo de primera vela: {type(candlesticks[0])}")
                print(f"Contenido de primera vela: {candlesticks[0]}")
            return {
                'ratio': 1.0,
                'is_significant': False,
                'is_increasing': False,
                'average': 0,
                'buy_pressure': 0
            }

    def _analyze_momentum(self, candlesticks: List[Dict]) -> Dict:
        """Analiza el momentum del precio"""
        try:
            closes = [float(candle['close']) for candle in candlesticks]

            if len(closes) < 42:
                return {
                    'short_term': 0,
                    'medium_term': 0,
                    'is_strong': False,
                    'is_positive': False,
                    'acceleration': 0
                }

            # Calcular cambios porcentuales
            change_24h = ((closes[-1] - closes[-6]) / closes[-6]) * 100
            change_7d = ((closes[-1] - closes[-42]) / closes[-42]) * 100

            # Calcular aceleración del momentum
            recent_changes = [
                ((closes[i] - closes[i-1]) / closes[i-1]) * 100
                for i in range(-5, 0)
            ]
            acceleration = sum(recent_changes) / len(recent_changes)

            return {
                'short_term': change_24h,
                'medium_term': change_7d,
                'is_strong': abs(change_24h) > self.thresholds['momentum']['strong_buy'],
                'is_positive': change_24h > 0 and change_7d > 0,
                'acceleration': acceleration
            }

        except Exception:
            return {
                'short_term': 0,
                'medium_term': 0,
                'is_strong': False,
                'is_positive': False,
                'acceleration': 0
            }

    # Mantener otros métodos existentes y agregar nuevos
    def _analyze_multiple_timeframes(self, market_data: Dict) -> Dict:
        mtf_analysis = {}

        for timeframe, data in market_data.items():
            try:
                closes = [float(candle['close']) for candle in data]
                volumes = [float(candle['volume']) for candle in data]

                # Calcular EMAs
                emas = {
                    period: self._calculate_ema(closes, period)
                    for period in self.indicators["ema"]
                }

                # Calcular RSI
                rsis = {
                    period: self._calculate_rsi(data[-period:])
                    for period in self.indicators["rsi"]
                }

                # Calcular Bandas de Bollinger
                for period in self.indicators["bb_periods"]:
                    for std in self.indicators["bb_std"]:
                        bb = self._calculate_bollinger_bands(closes, period, std)
                        key = f"bb_{period}_{std}"
                        mtf_analysis[f"{timeframe}_{key}"] = bb

                # MACD
                macd = self._calculate_macd(
                    closes,
                    self.indicators["macd"]["fast"],
                    self.indicators["macd"]["slow"],
                    self.indicators["macd"]["signal"]
                )

                mtf_analysis[timeframe] = {
                    'emas': emas,
                    'rsis': rsis,
                    'macd': macd,
                    'volume_trend': self._analyze_volume_trend(volumes),
                    'price_trend': self._analyze_price_trend(closes)
                }

            except Exception as e:
                print(ConsoleColors.warning(f"Error en análisis de {timeframe}: {str(e)}"))
                continue

        return mtf_analysis

    def _analyze_price_patterns(self, candlesticks: List[Dict]) -> Dict:
        """Identifica patrones de precio avanzados"""
        try:
            closes = [float(candle['close']) for candle in candlesticks]
            highs = [float(candle['high']) for candle in candlesticks]
            lows = [float(candle['low']) for candle in candlesticks]
            opens = [float(candle['open']) for candle in candlesticks]

            patterns = []

            # Doble fondo
            if (lows[-3] > lows[-2] and lows[-2] < lows[-1] and
                abs(lows[-3] - lows[-1]) < (highs[-2] - lows[-2]) * 0.1):
                patterns.append({
                    'type': 'bullish',
                    'name': 'Doble Fondo',
                    'reliability': 0.8
                })

            # Doble techo
            if (highs[-3] < highs[-2] and highs[-2] > highs[-1] and
                abs(highs[-3] - highs[-1]) < (highs[-2] - lows[-2]) * 0.1):
                patterns.append({
                    'type': 'bearish',
                    'name': 'Doble Techo',
                    'reliability': 0.8
                })

            # Martillo alcista
            for i in range(-3, 0):
                body = abs(opens[i] - closes[i])
                shadow_lower = min(opens[i], closes[i]) - lows[i]
                shadow_upper = highs[i] - max(opens[i], closes[i])
                if (shadow_lower > body * 2 and shadow_upper < body * 0.5 and
                    closes[i] > opens[i]):
                    patterns.append({
                        'type': 'bullish',
                        'name': 'Martillo Alcista',
                        'reliability': 0.7
                    })

            # Pin Bar (Price Action)
            for i in range(-3, 0):
                total_length = highs[i] - lows[i]
                body = abs(opens[i] - closes[i])
                if total_length > 0:
                    body_ratio = body / total_length
                    if body_ratio < 0.3:  # Cuerpo pequeño
                        upper_wick = highs[i] - max(opens[i], closes[i])
                        lower_wick = min(opens[i], closes[i]) - lows[i]
                        if upper_wick > 2 * body and lower_wick < 0.2 * total_length:
                            patterns.append({
                                'type': 'bearish',
                                'name': 'Pin Bar Bajista',
                                'reliability': 0.75
                            })
                        if lower_wick > 2 * body and upper_wick < 0.2 * total_length:
                            patterns.append({
                                'type': 'bullish',
                                'name': 'Pin Bar Alcista',
                                'reliability': 0.75
                            })

            return {
                'patterns': patterns,
                'pattern_count': len(patterns),
                'dominant_bias': 'bullish' if sum(1 for p in patterns if p['type'] == 'bullish') > \
                                           sum(1 for p in patterns if p['type'] == 'bearish') else 'bearish'
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error en análisis de patrones: {str(e)}"))
            return {'patterns': [], 'pattern_count': 0, 'dominant_bias': 'neutral'}

    def _analyze_market_correlation(self, symbol: str) -> Dict:
        """Analiza la correlación con BTC y el mercado general"""
        try:
            # Obtener datos de BTC y el par analizado
            btc_data = self.client.get_klines("BTCUSDT", interval='1h', limit=168)
            symbol_data = self.client.get_klines(symbol, interval='1h', limit=168)

            # Calcular correlación
            btc_returns = self._calculate_returns([float(candle['close']) for candle in btc_data])
            symbol_returns = self._calculate_returns([float(candle['close']) for candle in symbol_data])

            correlation = np.corrcoef(btc_returns, symbol_returns)[0, 1]

            # Calcular beta (volatilidad relativa a BTC)
            beta = np.std(symbol_returns) / np.std(btc_returns)

            # Analizar fuerza de mercado
            market_strength = self._analyze_market_strength()

            return {
                "correlation": correlation,
                "beta": beta,
                "btc_influence": "high" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.4 else "low",
                "market_strength": market_strength,
                "risk_level": "high" if beta > 1.5 else "moderate" if beta > 1 else "low"
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error en análisis de correlación: {str(e)}"))
            return {"correlation": 0, "beta": 1, "btc_influence": "unknown", "market_strength": "neutral", "risk_level": "moderate"}

    def _analyze_market_strength(self) -> str:
        """Analiza la fuerza general del mercado"""
        try:
            # Obtener datos de BTC
            btc_data = self.client.get_klines("BTCUSDT", interval='1d', limit=14)
            if not btc_data:
                return "neutral"

            closes = [float(candle['close']) for candle in btc_data]
            volumes = [float(candle['volume']) for candle in btc_data]

            # Analizar tendencia de BTC
            trend = self._analyze_trend(btc_data)

            # Analizar volumen
            avg_volume = sum(volumes[:-7]) / 7
            recent_volume = sum(volumes[-7:]) / 7
            volume_trend = "increasing" if recent_volume > avg_volume else "decreasing"

            # Determinar fuerza del mercado
            if trend == MarketTrend.STRONG_UPTREND and volume_trend == "increasing":
                return "very_strong"
            elif trend in [MarketTrend.STRONG_UPTREND, MarketTrend.UPTREND]:
                return "strong"
            elif trend in [MarketTrend.STRONG_DOWNTREND, MarketTrend.DOWNTREND]:
                return "weak"
            else:
                return "neutral"

        except Exception as e:
            print(ConsoleColors.error(f"Error en análisis de fuerza de mercado: {str(e)}"))
            return "neutral"

    def _combine_all_analysis(self, trend: MarketTrend, volume_analysis: Dict,
                            momentum: Dict, rsi: float, mtf_analysis: Dict,
                            pattern_analysis: Dict, correlation: Dict) -> Dict:
        """Combina todos los análisis para generar un análisis final"""
        try:
            # Calcular score base
            base_score = 0
            confidence = 0
            signals = []

            # Análisis de tendencia
            if trend in [MarketTrend.STRONG_UPTREND, MarketTrend.UPTREND]:
                base_score += self.weights["trend"]
                signals.append("Tendencia alcista")
                confidence += 0.2
            elif trend in [MarketTrend.STRONG_DOWNTREND, MarketTrend.DOWNTREND]:
                base_score -= self.weights["trend"]
                signals.append("Tendencia bajista")
                confidence += 0.2

            # Análisis de volumen
            if volume_analysis['is_significant'] and volume_analysis['is_increasing']:
                base_score += self.weights["volume"]
                signals.append("Volumen significativo y creciente")
                confidence += 0.15

            # Análisis de momentum
            if momentum['is_strong'] and momentum['is_positive']:
                base_score += self.weights["momentum"]
                signals.append("Momentum fuerte y positivo")
                confidence += 0.15
            elif momentum['is_strong'] and not momentum['is_positive']:
                base_score -= self.weights["momentum"]
                signals.append("Momentum fuerte pero negativo")
                confidence += 0.15

            # Análisis de patrones
            for pattern in pattern_analysis.get('patterns', []):
                if pattern['type'] == 'bullish':
                    base_score += self.weights["pattern"] * pattern['reliability']
                    signals.append(f"Patrón alcista: {pattern['name']}")
                    confidence += 0.1
                else:
                    base_score -= self.weights["pattern"] * pattern['reliability']
                    signals.append(f"Patrón bajista: {pattern['name']}")
                    confidence += 0.1

            # Análisis de correlación
            if correlation['market_strength'] == "very_strong":
                base_score *= 1.2
                confidence += 0.1
            elif correlation['market_strength'] == "weak":
                base_score *= 0.8
                confidence += 0.1

            return {
                'final_score': base_score,
                'confidence': min(confidence, 1.0),
                'signals': signals,
                'market_context': {
                    'trend': trend.value,
                    'volume_context': volume_analysis,
                    'momentum_context': momentum,
                    'correlation_context': correlation
                }
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error combinando análisis: {str(e)}"))
            return {
                'final_score': 0,
                'confidence': 0,
                'signals': ["Error en análisis"],
                'market_context': {}
            }

    def _calculate_returns(self, prices: List[float]) -> List[float]:
        """Calcula los retornos porcentuales"""
        return [(prices[i] - prices[i-1])/prices[i-1] for i in range(1, len(prices))]

    def _calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: float = 2.0) -> Dict:
        """Calcula las bandas de Bollinger"""
        try:
            if len(prices) < period:
                return {'upper': prices[-1], 'middle': prices[-1], 'lower': prices[-1]}

            # Calcular SMA y desviación estándar
            sma = sum(prices[-period:]) / period
            std = np.std(prices[-period:])

            return {
                'upper': sma + (std_dev * std),
                'middle': sma,
                'lower': sma - (std_dev * std)
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando Bollinger Bands: {str(e)}"))
            return {'upper': prices[-1], 'middle': prices[-1], 'lower': prices[-1]}

    def _calculate_macd(self, prices: List[float], fast_period: int = 12,
                       slow_period: int = 26, signal_period: int = 9) -> Dict:
        """Calcula el MACD (Moving Average Convergence Divergence)"""
        try:
            # Calcular EMAs
            fast_ema = self._calculate_ema(prices, fast_period)
            slow_ema = self._calculate_ema(prices, slow_period)

            # Calcular línea MACD
            macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]

            # Calcular línea de señal
            signal_line = self._calculate_ema(macd_line, signal_period)

            # Calcular histograma
            histogram = [m - s for m, s in zip(macd_line, signal_line)]

            return {
                'macd': macd_line[-1],
                'signal': signal_line[-1],
                'histogram': histogram[-1],
                'trending_up': macd_line[-1] > signal_line[-1],
                'momentum_strength': abs(histogram[-1]),
                'crossover': 'bullish' if macd_line[-1] > signal_line[-1] and macd_line[-2] <= signal_line[-2]
                            else 'bearish' if macd_line[-1] < signal_line[-1] and macd_line[-2] >= signal_line[-2]
                            else 'none'
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando MACD: {str(e)}"))
            return {'macd': 0, 'signal': 0, 'histogram': 0, 'trending_up': False,
                    'momentum_strength': 0, 'crossover': 'none'}



    def analyze_entry_timing(self, symbol: str) -> TimingWindow:
        """
        Analiza el mejor momento para entrar al mercado
        """
        try:
            # Obtener datos
            candlesticks = self.client.get_klines(symbol, interval='1h', limit=100)
            current_price = float(self.client.get_ticker_price(symbol)['price'])

            # Análisis técnico
            technical = self._analyze_technical_indicators(candlesticks)
            support_resistance = self._calculate_support_resistance(candlesticks)
            volatility = self._calculate_volatility(candlesticks)

            # Inicializar variables
            timing = EntryTiming.NOT_RECOMMENDED
            timeframe = "N/A"
            target_price = current_price
            confidence = 0.0
            conditions = []

            # Verificar RSI
            rsi = technical.get('rsi', 50)
            if rsi <= 30:
                timing = EntryTiming.IMMEDIATE
                timeframe = "0-4 horas"
                confidence = 0.8
                conditions.append(f"RSI en sobreventa ({rsi:.1f})")
            elif rsi >= 70:
                timing = EntryTiming.WAIT_DIP
                timeframe = "12-24 horas"
                confidence = 0.6
                conditions.append(f"RSI en sobrecompra ({rsi:.1f})")

            # Verificar niveles de soporte/resistencia
            if support_resistance:
                support = support_resistance.get('support')
                resistance = support_resistance.get('resistance')

                if support and resistance:
                    # Precio cerca del soporte
                    if current_price < support * 1.02:
                        timing = EntryTiming.IMMEDIATE
                        timeframe = "0-4 horas"
                        target_price = support * 1.01
                        confidence = 0.7
                        conditions.append(f"Precio cerca del soporte (${support:,.2f})")

                    # Precio cerca de la resistencia
                    elif current_price > resistance * 0.98:
                        timing = EntryTiming.WAIT_BREAKOUT
                        timeframe = "6-12 horas"
                        target_price = resistance * 1.02
                        confidence = 0.6
                        conditions.append(f"Precio cerca de resistencia (${resistance:,.2f})")

                    else:
                        timing = EntryTiming.WAIT_CONSOLIDATION
                        timeframe = "4-12 horas"
                        confidence = 0.4

            # Verificar volatilidad
            if volatility > 0.05:
                conditions.append(f"Alta volatilidad ({volatility:.1%})")
                confidence *= 0.8
            else:
                conditions.append(f"Baja volatilidad ({volatility:.1%})")

            return TimingWindow(
                timing=timing,
                timeframe=timeframe,
                target_price=target_price,
                confidence=confidence,
                conditions=conditions
            )

        except Exception as e:
            print(f"Error en determine_timing_window: {str(e)}")
            return TimingWindow(
                timing=EntryTiming.NOT_RECOMMENDED,
                timeframe="N/A",
                conditions=["Error en análisis"]
            )


    def _calculate_volatility(self, candlesticks: List[Dict], period: int = 14) -> float:
        """Calcula la volatilidad usando True Range"""
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
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]),
                    abs(lows[i] - closes[i-1])
                )
                true_ranges.append(true_range)

            # Calcular ATR
            atr = sum(true_ranges[-period:]) / period if true_ranges else 0

            # Calcular cambios porcentuales
            daily_returns = [
                (closes[i] - closes[i-1]) / closes[i-1]
                for i in range(1, len(closes))
            ]

            # Calcular desviación estándar
            std_dev = np.std(daily_returns) if daily_returns else 0

            # Combinar ATR y desviación estándar
            volatility = (atr / closes[-1] + std_dev) / 2

            return volatility

        except Exception as e:
            print(ConsoleColors.warning(f"Error calculando volatilidad: {str(e)}"))
            return 0.0


    def _calculate_ema(self, values: List[float], period: int) -> List[float]:
        """Calcula EMA (Exponential Moving Average)"""
        if not values:
            return []

        ema = [values[0]]  # Primer valor es el mismo
        multiplier = 2 / (period + 1)

        for price in values[1:]:
            ema.append((price * multiplier) + (ema[-1] * (1 - multiplier)))

        return ema

    def _generate_recommendation(self, trend: MarketTrend, volume_analysis: Dict,
                               momentum: Dict, rsi: float, timing: EntryTiming) -> Tuple[TradingSignal, SignalStrength, List[str]]:
        """Genera una recomendación de trading basada en el análisis y timing"""
        reasons = []
        score = 0

        # Si el timing no es favorable, ajustar la señal
        if timing in [EntryTiming.WAIT_DIP, EntryTiming.WAIT_BREAKOUT, EntryTiming.WAIT_CONSOLIDATION]:
            return TradingSignal.HOLD, SignalStrength.MODERATE, ["Esperando mejor punto de entrada"]

        # Análisis de tendencia
        if trend == MarketTrend.STRONG_UPTREND:
            score += 2
            reasons.append(f"Fuerte tendencia alcista")
        elif trend == MarketTrend.UPTREND:
            score += 1
            reasons.append(f"Tendencia alcista")
        elif trend == MarketTrend.STRONG_DOWNTREND:
            score -= 2
            reasons.append(f"Fuerte tendencia bajista")
        elif trend == MarketTrend.DOWNTREND:
            score -= 1
            reasons.append(f"Tendencia bajista")

        # Resto del análisis...
        # ... (mantener el resto de la lógica)

        return signal, strength, reasons

    def _calculate_support_resistance(self, candlesticks: List[Dict]) -> Dict:
        """Calcula niveles de soporte y resistencia"""
        try:
            closes = [float(candle['close']) for candle in candlesticks]
            highs = [float(candle['high']) for candle in candlesticks]
            lows = [float(candle['low']) for candle in candlesticks]

            # Usar pivots para identificar niveles
            window = 20
            supports = []
            resistances = []

            for i in range(window, len(candlesticks) - window):
                # Identificar pivots
                if min(lows[i-window:i+window]) == lows[i]:
                    supports.append(lows[i])
                if max(highs[i-window:i+window]) == highs[i]:
                    resistances.append(highs[i])

            # Calcular niveles finales
            support = max(supports[-3:]) if supports else min(lows)
            resistance = min(resistances[-3:]) if resistances else max(highs)

            return {
                'support': support,
                'resistance': resistance,
                'support_levels': sorted(supports[-3:]),
                'resistance_levels': sorted(resistances[-3:], reverse=True)
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando S/R: {str(e)}"))
            return {'support': 0, 'resistance': 0}

    def _analyze_volume_trend(self, volumes: List[float]) -> Dict:
        """Analiza la tendencia del volumen"""
        try:
            if len(volumes) < 20:
                return {'trend': 'neutral', 'strength': 0}

            short_ma = sum(volumes[-5:]) / 5
            long_ma = sum(volumes[-20:]) / 20

            trend = 'increasing' if short_ma > long_ma else 'decreasing'
            strength = abs((short_ma - long_ma) / long_ma)

            return {
                'trend': trend,
                'strength': strength,
                'short_ma': short_ma,
                'long_ma': long_ma
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error en análisis de tendencia de volumen: {str(e)}"))
            return {'trend': 'neutral', 'strength': 0}

    def _analyze_price_trend(self, closes: List[float]) -> Dict:
        """Analiza la tendencia del precio"""
        try:
            if len(closes) < 20:
                return {'trend': 'neutral', 'strength': 0}

            ema_short = self._calculate_ema(closes, 7)
            ema_long = self._calculate_ema(closes, 21)

            trend = 'bullish' if ema_short[-1] > ema_long[-1] else 'bearish'
            strength = abs((ema_short[-1] - ema_long[-1]) / ema_long[-1])

            return {
                'trend': trend,
                'strength': strength,
                'ema_short': ema_short[-1],
                'ema_long': ema_long[-1]
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error en análisis de tendencia de precio: {str(e)}"))
            return {'trend': 'neutral', 'strength': 0}

    def _identify_pattern(self, candlesticks: List[Dict]) -> Dict:
        """Identifica patrones de velas"""
        try:
            opens = [float(candle['open']) for candle in candlesticks[-5:]]
            closes = [float(candle['close']) for candle in candlesticks[-5:]]
            highs = [float(candle['high']) for candle in candlesticks[-5:]]
            lows = [float(candle['low']) for candle in candlesticks[-5:]]

            patterns = []

            # Doji
            if abs(opens[-1] - closes[-1]) <= (highs[-1] - lows[-1]) * 0.1:
                patterns.append({'name': 'Doji', 'type': 'neutral'})

            # Martillo
            body = abs(opens[-1] - closes[-1])
            lower_wick = min(opens[-1], closes[-1]) - lows[-1]
            upper_wick = highs[-1] - max(opens[-1], closes[-1])

            if lower_wick > body * 2 and upper_wick < body * 0.5:
                patterns.append({'name': 'Hammer', 'type': 'bullish'})

            # Estrella fugaz
            if upper_wick > body * 2 and lower_wick < body * 0.5:
                patterns.append({'name': 'Shooting Star', 'type': 'bearish'})

            return {
                'patterns': patterns,
                'count': len(patterns),
                'bias': patterns[0]['type'] if patterns else 'neutral'
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error identificando patrones: {str(e)}"))
            return {'patterns': [], 'count': 0, 'bias': 'neutral'}

    def _calculate_trade_levels(self, trend_analysis: Dict, technical_analysis: Dict,
                              market_conditions: Dict, current_price: float) -> Dict:
        """Calcula niveles de entrada y salida precisos"""
        try:
            # Calcular ATR para stops dinámicos
            atr = self._calculate_atr(market_conditions['data'])

            # Encontrar niveles clave
            support_resistance = self._calculate_support_resistance(market_conditions['data'])

            # Calcular entrada
            if trend_analysis['trend'] == 'bullish':
                entry = max(
                    current_price,  # No entrar arriba del precio actual
                    support_resistance['support'] * 1.01  # 1% sobre soporte
                )

                # Stop loss basado en ATR y soporte
                stop_loss = max(
                    entry - (atr * 2),
                    support_resistance['support'] * 0.99  # 1% bajo soporte
                )

                # Take profit en dos niveles
                tp1 = entry + ((entry - stop_loss) * self.strategy_params["risk_params"]["partial_tp_ratio"])
                tp2 = min(
                    entry + (atr * 4),
                    support_resistance['resistance'] * 0.99  # 1% bajo resistencia
                )
            else:
                return None

            return {
                'entry': entry,
                'stop_loss': stop_loss,
                'take_profit_1': tp1,
                'take_profit_2': tp2,
                'risk_reward_ratio': (tp2 - entry) / (entry - stop_loss)
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando niveles: {str(e)}"))
            return None

    def _calculate_atr(self, candlesticks: List[Dict], period: int = 14) -> float:
        """Calcula el ATR (Average True Range)"""
        try:
            if not candlesticks or len(candlesticks) < period:
                return 0.0

            true_ranges = []
            for i in range(1, len(candlesticks)):
                high = float(candlesticks[i]['high'])
                low = float(candlesticks[i]['low'])
                prev_close = float(candlesticks[i-1]['close'])

                tr = max([
                    high - low,  # Rango actual
                    abs(high - prev_close),  # Movimiento desde el cierre anterior al máximo
                    abs(low - prev_close)  # Movimiento desde el cierre anterior al mínimo
                ])
                true_ranges.append(tr)

            return sum(true_ranges[-period:]) / period

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando ATR: {str(e)}"))
            return 0.0

    def _determine_timing(self, current_price: float, rsi: float, volume_analysis: Dict,
                         support_resistance: Dict, volatility: float, pattern: Dict) -> Tuple[EntryTiming, str, float, float, List[str]]:
        """Determina el mejor momento para entrar"""
        conditions = []
        confidence = 0.0
        timing = EntryTiming.NOT_RECOMMENDED
        timeframe = "12-24 horas"  # timeframe por defecto
        target_price = current_price

        try:
            # Ajustar timeframes basados en volatilidad
            high_volatility = volatility > self.timing_thresholds["high_volatility"]
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

            if support > 0:
                distance_to_support = (current_price - support) / support
                if distance_to_support <= self.timing_thresholds["price_support"]:
                    timing = EntryTiming.IMMEDIATE
                    timeframe = "0-4 horas"
                    target_price = support
                    confidence += 0.25
                    conditions.append(f"Precio cerca del soporte (${support:,.8f})")

            if resistance > current_price:
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
            if pattern and pattern.get('patterns'):
                pattern_type = pattern['patterns'][0]['type']
                pattern_name = pattern['patterns'][0]['name']
                if pattern_type == 'bullish':
                    if timing != EntryTiming.IMMEDIATE:
                        timing = EntryTiming.WAIT_BREAKOUT
                        timeframe = "4-8 horas"
                    confidence += 0.2
                    conditions.append(f"Patrón alcista: {pattern_name}")
                elif pattern_type == 'bearish':
                    timing = EntryTiming.WAIT_DIP
                    timeframe = "12-24 horas"
                    confidence -= 0.1
                    conditions.append(f"Patrón bajista: {pattern_name}")

            # Ajustar confianza final
            confidence = min(max(confidence, 0.0), 1.0)

        except Exception as e:
            print(ConsoleColors.error(f"Error en determine_timing: {str(e)}"))
            return EntryTiming.NOT_RECOMMENDED, "N/A", current_price, 0.0, ["Error en análisis"]

        return timing, timeframe, target_price, confidence, conditions

    def _adjust_timeframe(self, timeframe: str, multiplier: float) -> str:
        """Ajusta el timeframe según el multiplicador"""
        try:
            import re
            numbers = re.findall(r'\d+', timeframe)
            if len(numbers) == 2:
                min_time = int(float(numbers[0]) * multiplier)
                max_time = int(float(numbers[1]) * multiplier)
                return f"{min_time}-{max_time} horas"
            return timeframe
        except Exception:
            return timeframe


    def _calculate_optimal_entry(self, current_price, trend, volume, patterns, levels):
        """
        Calcula el punto óptimo de entrada basado en múltiples factores
        """
        try:
            if trend['trend'] != 'bullish':
                return {
                    'is_valid': False,
                    'optimal_entry': current_price,
                    'reason': 'Tendencia no favorable'
                }

            # Obtener niveles de soporte y resistencia
            support = levels.get('support', current_price * 0.95)
            resistance = levels.get('resistance', current_price * 1.05)

            # Calcular entrada óptima basada en niveles
            if current_price < support * 1.02:  # Cerca del soporte
                optimal_entry = support * 1.01
            elif current_price > resistance * 0.98:  # Cerca de la resistencia
                optimal_entry = None  # No entrar cerca de resistencia
            else:
                optimal_entry = current_price

            # Validar con volumen
            if volume.get('is_significant', False) and volume.get('is_increasing', False):
                confidence = 'high'
            else:
                confidence = 'low'

            # Validar con patrones
            if patterns.get('pattern_type') == 'bullish':
                confidence = 'high'
            elif patterns.get('pattern_type') == 'bearish':
                optimal_entry = None

            if optimal_entry is None:
                return {
                    'is_valid': False,
                    'optimal_entry': current_price,
                    'reason': 'No se encontró punto de entrada favorable'
                }

            return {
                'is_valid': True,
                'optimal_entry': optimal_entry,
                'reason': f'Entrada óptima encontrada con confianza {confidence}',
                'support': support,
                'resistance': resistance
            }

        except Exception as e:
            print(f"Error calculando entrada óptima: {str(e)}")
            return {
                'is_valid': False,
                'optimal_entry': current_price,
                'reason': 'Error en cálculo'
            }

    def _validate_volume(self, candlesticks: List[Dict]) -> bool:
        """
        Valida que el volumen sea suficiente y consistente
        """
        try:
            volumes = [float(candle['volume']) for candle in candlesticks[-20:]]
            avg_volume = sum(volumes) / len(volumes)
            current_volume = volumes[-1]

            # Volumen debe ser mayor al promedio
            if current_volume < avg_volume:
                return False

            # Volatilidad del volumen no debe ser excesiva
            vol_std = np.std(volumes)
            if vol_std / avg_volume > 2:  # Coeficiente de variación > 200%
                return False

            return True

        except Exception:
            return False

    def _generate_entry_reason(self, conditions: Dict, trend: MarketTrend,
                             rsi: float, volume_analysis: Dict) -> List[str]:
        """Genera razones detalladas para la entrada"""
        reasons = []

        if conditions['trend_alignment']:
            reasons.append("Alineación alcista de EMAs (8>21>50>200)")

        if conditions['rsi_condition']:
            reasons.append(f"RSI en zona óptima ({rsi:.1f})")

        if conditions['volume_condition']:
            reasons.append(f"Volumen significativo ({volume_analysis['ratio']:.1f}x promedio)")

        if conditions['volatility_condition']:
            reasons.append("Volatilidad controlada en rango óptimo")

        if conditions['price_position']:
            reasons.append("Precio por encima de EMA 200 (tendencia principal alcista)")

        if trend == MarketTrend.STRONG_UPTREND:
            reasons.append("Fuerte tendencia alcista confirmada")

        return reasons



    def _generate_technical_reason(self, macd: Dict, rsi: float, bb: Dict, price: float) -> str:
        """Genera razones detalladas del análisis técnico"""
        reasons = []

        if macd['crossover'] == 'bullish':
            reasons.append(f"Cruce alcista del MACD (Histograma: {macd['histogram']:.8f})")

        if rsi <= self.strategy_params["rsi_params"]["oversold"]:
            reasons.append(f"RSI en sobreventa ({rsi:.2f})")
        elif rsi >= self.strategy_params["rsi_params"]["overbought"]:
            reasons.append(f"RSI en sobrecompra ({rsi:.2f})")

        if price <= bb['lower']:
            reasons.append("Precio tocando banda inferior de Bollinger")
        elif price >= bb['upper']:
            reasons.append("Precio tocando banda superior de Bollinger")

        return " | ".join(reasons)

    def _generate_trade_recommendation(self, trend_analysis: Dict, technical_analysis: Dict,
                                     candle_patterns: Dict, trade_levels: Dict,
                                     market_conditions: Dict) -> TradeRecommendation:
        """Genera la recomendación final de trading"""
        try:
            if not trade_levels or trade_levels['risk_reward_ratio'] < 2:
                return self._generate_hold_recommendation(
                    "Ratio riesgo/beneficio insuficiente",
                    float(self.client.get_ticker_price(symbol)['price'])
                )

            reasons = [
                f"Tendencia alcista confirmada en múltiples timeframes",
                technical_analysis['reason'],
                f"Patrón de velas: {candle_patterns['pattern_name']}",
                f"Ratio Riesgo/Beneficio: 1:{trade_levels['risk_reward_ratio']:.1f}"
            ]

            return TradeRecommendation(
                signal=TradingSignal.BUY,
                strength=SignalStrength.STRONG,
                reasons=reasons,
                entry_price=trade_levels['entry'],
                stop_loss=trade_levels['stop_loss'],
                take_profit=trade_levels['take_profit_2']
            )

        except Exception as e:
            print(ConsoleColors.error(f"Error generando recomendación: {str(e)}"))
            return self._generate_hold_recommendation(str(e), current_price)

    def _analyze_candle_patterns(self, candlesticks: List[Dict]) -> Dict:
        """Analiza patrones de velas japonesas"""
        try:
            opens = [float(candle['open']) for candle in candlesticks[-5:]]
            closes = [float(candle['close']) for candle in candlesticks[-5:]]
            highs = [float(candle['high']) for candle in candlesticks[-5:]]
            lows = [float(candle['low']) for candle in candlesticks[-5:]]

            patterns = []

            # Patrón envolvente alcista
            if closes[-2] < opens[-2] and closes[-1] > opens[-1] and \
               opens[-1] < closes[-2] and closes[-1] > opens[-2]:
                patterns.append({
                    'name': 'Bullish Engulfing',
                    'type': 'bullish',
                    'strength': 0.8
                })

            # Patrón envolvente bajista
            if closes[-2] > opens[-2] and closes[-1] < opens[-1] and \
               opens[-1] > closes[-2] and closes[-1] < opens[-2]:
                patterns.append({
                    'name': 'Bearish Engulfing',
                    'type': 'bearish',
                    'strength': 0.8
                })

            # Martillo alcista
            body = abs(opens[-1] - closes[-1])
            lower_shadow = min(opens[-1], closes[-1]) - lows[-1]
            upper_shadow = highs[-1] - max(opens[-1], closes[-1])

            if lower_shadow > (2 * body) and upper_shadow < (0.5 * body):
                patterns.append({
                    'name': 'Hammer',
                    'type': 'bullish',
                    'strength': 0.7
                })

            # Otros patrones relevantes...

            return {
                'has_confirmation': len(patterns) > 0,
                'patterns': patterns,
                'pattern_name': patterns[0]['name'] if patterns else 'No Pattern',
                'pattern_type': patterns[0]['type'] if patterns else 'neutral',
                'pattern_strength': patterns[0]['strength'] if patterns else 0
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando patrones de velas: {str(e)}"))
            return {
                'has_confirmation': False,
                'patterns': [],
                'pattern_name': 'Error',
                'pattern_type': 'neutral',
                'pattern_strength': 0
            }

    def _generate_market_condition_reason(self, volatility: float,
                                        volume_analysis: Dict,
                                        momentum: Dict) -> str:
        """Genera razón detallada de las condiciones de mercado"""
        reasons = []

        if volatility > self.strategy_params["risk_params"]["volatility_threshold"]:
            reasons.append(f"Volatilidad muy alta ({volatility:.1%})")

        if volume_analysis['ratio'] < self.thresholds['volume']['significant']:
            reasons.append(f"Volumen insuficiente ({volume_analysis['ratio']:.1f}x promedio)")

        if momentum['is_strong'] and not momentum['is_positive']:
            reasons.append("Momentum fuertemente negativo")

        return " | ".join(reasons) if reasons else "Condiciones de mercado favorables"



    def _determine_signal_strength(self, trend_analysis, volume_analysis, pattern_analysis, entry_analysis):
        """
        Determina la señal y su fuerza basándose en múltiples factores
        """
        try:
            score = 0
            conditions_met = 0
            total_conditions = 4

            # 1. Análisis de tendencia (0-40 puntos)
            if trend_analysis.get('trend') == 'bullish':
                score += 40 * trend_analysis.get('strength', 0)
                conditions_met += 1
            elif trend_analysis.get('trend') == 'bearish':
                score -= 40 * trend_analysis.get('strength', 0)

            # 2. Análisis de volumen (0-30 puntos)
            if volume_analysis.get('is_significant', False):
                score += 15
                conditions_met += 1
            if volume_analysis.get('is_increasing', False):
                score += 15
                conditions_met += 1

            # 3. Patrones de precio (0-20 puntos)
            if pattern_analysis and pattern_analysis.get('patterns'):
                if pattern_analysis.get('pattern_type') == 'bullish':
                    score += 20 * pattern_analysis.get('pattern_strength', 0)
                    conditions_met += 1
                elif pattern_analysis.get('pattern_type') == 'bearish':
                    score -= 20 * pattern_analysis.get('pattern_strength', 0)

            # 4. Entrada óptima (0-10 puntos)
            if entry_analysis and entry_analysis.get('is_valid', False):
                score += 10
                conditions_met += 1

            # Calcular confianza basada en condiciones cumplidas
            confidence = conditions_met / total_conditions

            # Determinar señal y fuerza
            if score >= 70 and confidence >= 0.75:
                return TradingSignal.STRONG_BUY, SignalStrength.STRONG
            elif score >= 40 and confidence >= 0.5:
                return TradingSignal.BUY, SignalStrength.MODERATE
            elif score <= -70 and confidence >= 0.75:
                return TradingSignal.STRONG_SELL, SignalStrength.STRONG
            elif score <= -40 and confidence >= 0.5:
                return TradingSignal.SELL, SignalStrength.MODERATE
            else:
                return TradingSignal.HOLD, SignalStrength.WEAK

        except Exception as e:
            print(f"Error en determine_signal_strength: {str(e)}")
            return TradingSignal.HOLD, SignalStrength.WEAK

    def _generate_specific_reasons(self, trend_analysis, volume_analysis, pattern_analysis, entry_analysis):
        """
        Genera razones específicas basadas en el análisis
        """
        try:
            reasons = []

            # Razones de tendencia
            if trend_analysis.get('trend') == 'bullish':
                reasons.append(f"Tendencia alcista con fuerza {trend_analysis.get('strength', 0):.2%}")
            elif trend_analysis.get('trend') == 'bearish':
                reasons.append(f"Tendencia bajista con fuerza {trend_analysis.get('strength', 0):.2%}")

            # Razones de volumen
            if volume_analysis.get('is_significant'):
                reasons.append(f"Volumen significativo ({volume_analysis.get('ratio', 0):.1f}x promedio)")
            if volume_analysis.get('is_increasing'):
                reasons.append("Volumen creciente")

            # Razones de patrones
            if pattern_analysis and pattern_analysis.get('patterns'):
                reasons.append(f"Patrón {pattern_analysis.get('pattern_name', 'desconocido')} detectado")

            # Razones de entrada
            if entry_analysis.get('is_valid'):
                reasons.append(f"Punto de entrada óptimo identificado en ${entry_analysis.get('optimal_entry', 0):.8f}")
                if entry_analysis.get('reason'):
                    reasons.append(entry_analysis['reason'])
            else:
                reasons.append("No se encontró punto de entrada óptimo")

            return reasons if reasons else ["Análisis incompleto o sin señales claras"]

        except Exception as e:
            print(f"Error generando razones: {str(e)}")
            return ["Error en análisis"]

    def _determine_timing_window(self, current_price, technical, levels, volatility):
        """
        Determina la ventana de timing óptima
        """
        try:
            conditions = []

            # Analizar RSI
            rsi = technical.get('rsi', 50)
            if rsi < 30:
                timing = EntryTiming.IMMEDIATE
                timeframe = "0-4 horas"
                confidence = 0.8
                conditions.append(f"RSI en sobreventa ({rsi:.1f})")
            elif rsi > 70:
                timing = EntryTiming.WAIT_DIP
                timeframe = "12-24 horas"
                confidence = 0.6
                conditions.append(f"RSI en sobrecompra ({rsi:.1f})")

            # Analizar niveles
            support = levels.get('support')
            resistance = levels.get('resistance')

            if support and (current_price - support) / support < 0.02:
                if timing != EntryTiming.WAIT_DIP:
                    timing = EntryTiming.IMMEDIATE
                    timeframe = "0-4 horas"
                    confidence = 0.7
                conditions.append(f"Precio cerca del soporte (${support:.2f})")
                target_price = support * 1.01

            elif resistance and (resistance - current_price) / current_price < 0.02:
                timing = EntryTiming.WAIT_BREAKOUT
                timeframe = "6-12 horas"
                confidence = 0.6
                conditions.append(f"Precio cerca de resistencia (${resistance:.2f})")
                target_price = resistance * 1.02

            else:
                timing = EntryTiming.WAIT_CONSOLIDATION
                timeframe = "4-12 horas"
                confidence = 0.4
                target_price = current_price

            # Ajustar por volatilidad
            if volatility > 0.05:  # Alta volatilidad
                confidence *= 0.8
                conditions.append(f"Alta volatilidad ({volatility:.1%})")
            else:
                conditions.append(f"Baja volatilidad ({volatility:.1%})")

            return timing, timeframe, target_price, confidence, conditions

        except Exception as e:
            print(f"Error en determine_timing_window: {str(e)}")
            return EntryTiming.NOT_RECOMMENDED, "N/A", current_price, 0.0, ["Error en análisis"]

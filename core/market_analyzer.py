# core/market_analyzer.py
import numpy as np
from typing import Dict, List, Optional, Tuple
from utils.console_colors import ConsoleColors
from alerts.alert_manager import AlertManager
from models.data_classes import TradeRecommendation, TimingWindow
from models.enums import MarketTrend, TradingSignal, SignalStrength, EntryTiming
from core.coinmarketcap_client import CoinMarketCapClient
from config import config


class MarketAnalyzer:

    def __init__(self, client, alert_manager=None):
        self.client = client
        self.alert_manager = alert_manager
        self.cmc_client = CoinMarketCapClient(config.CMC_API_KEY)

        # Parámetros optimizados de la estrategia
        self.strategy_params = {
            # EMAs (Medias Móviles Exponenciales)
            "ema_periods": {
                "fast": 8,  # Señales rápidas de entrada/salida
                "medium": 21,  # Confirmación de tendencia a medio plazo
                "slow": 50,  # Confirmación de tendencia a largo plazo
                "trend": 200,  # Tendencia principal del mercado
                "validation_periods": [13, 34, 89],  # Fibonacci EMAs para confirmación
                "min_separation": 0.5,  # % mínimo de separación entre EMAs
                "alignment_threshold": 0.8,  # % de EMAs que deben estar alineadas
            },
            # MACD (Moving Average Convergence Divergence)
            "macd_params": {
                "fast": 12,  # EMA rápida para MACD
                "slow": 26,  # EMA lenta para MACD
                "signal": 9,  # Línea de señal
                # Optimizaciones adicionales
                "min_hist_strength": 0.2,  # Fuerza mínima del histograma
                "confirmation_periods": 3,  # Períodos de confirmación
                "divergence_threshold": 0.1,  # % para confirmar divergencias
                "volume_confirmation": True,  # Requerir confirmación de volumen
            },
            # Bandas de Bollinger
            "bollinger_params": {
                "period": 20,  # Período para cálculo
                "std_dev": 2,  # Desviaciones estándar
                # Optimizaciones adicionales
                "squeeze_threshold": 0.5,  # % para detectar compresión
                "expansion_threshold": 2.5,  # % para detectar expansión
                "touch_count": 2,  # Toques necesarios para validación
                "bandwidth_min": 1.0,  # Ancho mínimo de bandas
                "percent_b_threshold": 0.05,  # % B para confirmación
            },
            # RSI (Índice de Fuerza Relativa)
            "rsi_params": {
                "period": 14,  # Período de cálculo
                "oversold": 30,  # Nivel de sobreventa
                "overbought": 70,  # Nivel de sobrecompra
                # Optimizaciones adicionales
                "divergence_periods": 5,  # Períodos para buscar divergencias
                "trend_alignment": True,  # Requerir alineación con tendencia
                "confirmation_levels": {  # Niveles adicionales de confirmación
                    "extreme_oversold": 20,
                    "extreme_overbought": 80,
                    "neutral_low": 45,
                    "neutral_high": 55,
                },
                "momentum_threshold": 0.3,  # % para cambio de momentum
            },
            # Gestión de Riesgo y Toma de Ganancias
            "risk_params": {
                "max_risk_percent": 2.0,  # Riesgo máximo por operación
                "risk_reward_min": 2.0,  # Ratio mínimo riesgo/beneficio
                # Niveles de toma de ganancias parciales
                "take_profit_levels": [
                    {"size": 0.40, "ratio": 0.09},  # 40% de la posición a +2%
                    {"size": 0.30, "ratio": 0.08},  # 30% de la posición a +5%
                    {"size": 0.30, "ratio": 0.07},  # 30% de la posición a +10%
                ],
                # Configuración del stop loss
                "stop_loss": {
                    "initial": 0.02,  # 2% inicial
                    "breakeven": 0.005,  # Mover a breakeven en 0.5%
                    "trailing": {
                        "activation": 1.02,  # Activar trailing en +2%
                        "step": 0.005,  # Paso del trailing 0.5%
                    },
                },
                # Gestión de posición
                "position": {
                    "base_size": 1.0,  # Tamaño base de la posición
                    "scale_in": [0.5, 0.3, 0.2],  # Distribución de entradas
                    "max_trades": 3,  # Máximo de operaciones simultáneas
                },
                # Filtros de mercado
                "filters": {
                    "min_volume_24h": 1000000,  # Volumen mínimo en 24h
                    "min_liquidity_ratio": 0.02,  # Ratio mínimo de liquidez
                    "max_spread_percent": 0.1,  # Spread máximo permitido
                    "news_impact_delay": 30,  # Minutos de espera post-noticias
                },
            },
            # Análisis Técnico Avanzado
            "advanced_analysis": {
                "pattern_recognition": {
                    "min_pattern_size": 0.01,  # Tamaño mínimo del patrón
                    "confirmation_candles": 2,  # Velas de confirmación
                    "volume_confirmation": True,  # Requerir confirmación de volumen
                },
                "support_resistance": {
                    "pivot_periods": 20,  # Períodos para pivotes
                    "level_strength": 3,  # Toques para confirmar nivel
                    "zone_thickness": 0.002,  # Grosor de zonas S/R
                },
                "market_structure": {
                    "swing_threshold": 0.01,  # % para identificar swings
                    "trend_validation": 3,  # Pivotes para confirmar tendencia
                    "structure_break": 0.005,  # % para ruptura de estructura
                },
            },
            "volume": {
                "min_24h": 500000,  # Volumen mínimo en 24h
                "min_ratio": 1.5,  # Ratio mínimo respecto al promedio
            },
            "volatility": {
                "min": 0.02,  # Volatilidad mínima (2%)
                "max": 0.15,  # Volatilidad máxima (15%)
                "threshold": 0.08,  # Umbral para decisiones (8%)
            },
            "trend": {
                "min_strength": 0.3,  # Fuerza mínima de tendencia
                "confirmation_periods": 3,
            },
        }

        # Umbrales para detección de señales
        self.thresholds = {
            "momentum": {"strong_buy": 15, "buy": 5, "strong_sell": -15, "sell": -5},
            "volume": {"significant": 2, "moderate": 1.5, "low": 0.5},
            "rsi": {
                "strong_oversold": 20,
                "oversold": 30,
                "overbought": 70,
                "strong_overbought": 80,
            },
        }

        # Umbrales para timing de entrada/salida
        self.signal_thresholds = {
            "score": {
                "strong_buy": 70,  # Reducido de 80 a 70
                "buy": 55,  # Reducido de 65 a 55
                "strong_sell": -70,
                "sell": -55,
            },
            "confirmations": {
                "strong": 5,  # Reducido de 6 a 5
                "moderate": 3,  # Reducido de 4 a 3
                "weak": 2,
            },
        }

        # Umbrales de mercado generales
        self.market_thresholds = (
            {
                "profit_lock": 0.01,  # Asegurar ganancias después de 1%
                "trend_strength": 0.5,  # Reducido de 0.6 a 0.5
                "volume_confirm": 1.3,  # Reducido de 1.5 a 1.3
                "exit_conditions": {
                    "rsi_overbought": 80,  # Aumentado de 75 a 80
                    "trend_reversal": 0.25,  # Reducido de 0.3 a 0.25
                    "volume_drop": 0.4,  # Reducido de 0.5 a 0.4
                },
            },
        )
        self.analysis_weights = {
            "technical": 0.35,  # Análisis técnico
            "market": 0.25,  # Datos de mercado (CMC)
            "volume": 0.20,  # Análisis de volumen
            "momentum": 0.20,  # Momentum y tendencia
        }
        self.take_profit_params = {
            "base_levels": [
                {"level": 1.02, "size": 0.3},  # +2% - 30% de la posición
                {"level": 1.035, "size": 0.3},  # +3.5% - 30% de la posición
                {"level": 1.05, "size": 0.4},  # +5% - 40% de la posición
            ],
            "momentum_multiplier": {
                "strong": 1.5,  # Aumentar targets en momentum fuerte
                "moderate": 1.2,  # Aumentar moderadamente
                "weak": 1.0,  # Mantener targets base
            },
            "volatility_adjustment": {
                "high": 1.3,  # Ampliar targets en alta volatilidad
                "normal": 1.0,  # Mantener targets normales
                "low": 0.8,  # Reducir targets en baja volatilidad
            },
        }

    def _validate_market_conditions(self, data: Dict) -> bool:
        """Validación avanzada de condiciones de mercado"""
        try:
            # Validar volumen
            if data.get("volume", 0) < self.strategy_params["volume"]["min_24h"]:
                return False

            # Validar volatilidad
            volatility = data.get("volatility", 0)
            if not (
                self.strategy_params["volatility"]["min"]
                <= volatility
                <= self.strategy_params["volatility"]["max"]
            ):
                return False

            # Validar tendencia
            trend_data = data.get("trend", {})
            if not trend_data:
                return False

            trend_strength = trend_data.get("strength", 0)
            if trend_strength < self.strategy_params["trend"]["min_strength"]:
                return False

            return True

        except Exception as e:
            print(
                ConsoleColors.error(f"Error validando condiciones de mercado: {str(e)}")
            )
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
                volatility
                <= self.strategy_params["risk_params"]["volatility_threshold"]
                and volume_analysis["ratio"] >= self.thresholds["volume"]["significant"]
                and self._validate_volume(candlesticks)
            )

            return {
                "is_tradeable": is_tradeable,
                "reason": self._generate_market_condition_reason(
                    volatility, volume_analysis, momentum
                ),
                "data": candlesticks,
                "volatility": volatility,
                "volume": volume_analysis,
                "momentum": momentum,
            }

        except Exception as e:
            print(
                ConsoleColors.error(
                    f"Error analizando condiciones de mercado: {str(e)}"
                )
            )
            return {
                "is_tradeable": False,
                "reason": "Error en análisis de condiciones de mercado",
                "data": candlesticks,
            }

    def _extract_validated_candle_data(self, candlesticks: List[Dict]) -> Dict:
        """Extrae y valida datos de las velas"""
        try:
            closes = []
            highs = []
            lows = []
            volumes = []
            timestamps = []

            for candle in candlesticks:
                try:
                    if isinstance(candle, (list, tuple)):
                        closes.append(float(candle[4]))
                        highs.append(float(candle[2]))
                        lows.append(float(candle[3]))
                        volumes.append(float(candle[5]))
                        timestamps.append(int(candle[0]))
                    elif isinstance(candle, dict):
                        closes.append(float(candle["close"]))
                        highs.append(float(candle["high"]))
                        lows.append(float(candle["low"]))
                        volumes.append(float(candle["volume"]))
                        timestamps.append(int(candle["timestamp"]))
                except (ValueError, KeyError, IndexError) as e:
                    continue

            # Validar calidad de datos
            if len(closes) < 200:
                return {"is_valid": False, "error": "Insufficient data"}

            # Validar continuidad temporal
            if not self._validate_time_continuity(timestamps):
                return {"is_valid": False, "error": "Time gaps detected"}

            # Validar coherencia de precios
            if not self._validate_price_coherence(highs, lows, closes):
                return {"is_valid": False, "error": "Price coherence issues"}

            return {
                "is_valid": True,
                "closes": closes,
                "highs": highs,
                "lows": lows,
                "volumes": volumes,
                "timestamps": timestamps,
            }

        except Exception as e:
            return {"is_valid": False, "error": str(e)}

    def _calculate_advanced_rsi(self, timeframes: Dict) -> Dict:
        """Cálculo avanzado de RSI con múltiples timeframes"""
        try:
            rsi_data = {}
            weighted_rsi = 0
            total_weight = 0

            for tf_name, tf_info in timeframes.items():
                rsi = self._calculate_rsi(tf_info["data"])
                rsi_data[tf_name] = {
                    "value": rsi,
                    "zone": self._get_rsi_zone(rsi),
                    "momentum": self._get_rsi_momentum(rsi),
                }
                weighted_rsi += rsi * tf_info["weight"]
                total_weight += tf_info["weight"]

            # Añadir análisis de divergencias
            rsi_data["divergences"] = self._analyze_rsi_divergences(timeframes)
            rsi_data["weighted_value"] = (
                weighted_rsi / total_weight if total_weight > 0 else 50
            )
            rsi_data["trend"] = self._get_rsi_trend(rsi_data)
            rsi_data["strength"] = self._calculate_rsi_strength(rsi_data)

            return rsi_data

        except Exception as e:
            print(f"Error en RSI avanzado: {str(e)}")
            return {"value": 50, "trend": "neutral", "strength": 0}

    def _get_rsi_zone(self, rsi: float) -> str:
        """Determina la zona del RSI con más detalle"""
        if rsi >= 80:
            return "extreme_overbought"
        if rsi >= 70:
            return "overbought"
        if rsi >= 60:
            return "bullish"
        if rsi <= 20:
            return "extreme_oversold"
        if rsi <= 30:
            return "oversold"
        if rsi <= 40:
            return "bearish"
        return "neutral"

    def _get_rsi_momentum(self, rsi: float) -> float:
        """Calcula el momentum del RSI"""
        try:
            rsi_changes = [rsi[i] - rsi[i - 1] for i in range(1, len(rsi))]
            return sum(rsi_changes[-3:]) / 3  # Promedio de los últimos 3 cambios
        except Exception:
            return 0

    def _analyze_rsi_divergences(self, timeframes: Dict) -> Dict:
        """Analiza divergencias en el RSI"""
        divergences = {
            "bullish": False,
            "bearish": False,
            "strength": 0,
            "timeframe": None,
        }

        try:
            for tf_name, tf_info in timeframes.items():
                prices = tf_info["data"]
                rsi_values = [
                    self._calculate_rsi(prices[i:]) for i in range(len(prices) - 14)
                ]

                # Buscar divergencias
                if len(prices) >= 5 and len(rsi_values) >= 5:
                    price_trend = prices[-1] < prices[-5]
                    rsi_trend = rsi_values[-1] > rsi_values[-5]

                    if price_trend and not rsi_trend:
                        divergences["bullish"] = True
                        divergences["timeframe"] = tf_name
                        divergences["strength"] = (
                            abs(prices[-1] - prices[-5]) / prices[-5]
                        )
                    elif not price_trend and rsi_trend:
                        divergences["bearish"] = True
                        divergences["timeframe"] = tf_name
                        divergences["strength"] = (
                            abs(prices[-1] - prices[-5]) / prices[-5]
                        )

        except Exception as e:
            print(f"Error en análisis de divergencias RSI: {str(e)}")

        return divergences

    def _calculate_technical_confidence(self, indicators: Dict) -> float:
        """Calcula la confianza general del análisis técnico"""
        try:
            confidence = 0.0
            weights = {
                "trend": 0.3,
                "consistency": 0.3,
                "divergences": 0.2,
                "momentum": 0.2,
            }

            # Evaluar tendencia
            if indicators["trend"]["strength"] > 0.7:
                confidence += weights["trend"]
            elif indicators["trend"]["strength"] > 0.5:
                confidence += weights["trend"] * 0.7

            # Evaluar consistencia
            if indicators["consistency"]["score"] > 0.7:
                confidence += weights["consistency"]
            elif indicators["consistency"]["score"] > 0.5:
                confidence += weights["consistency"] * 0.7

            # Evaluar divergencias
            if indicators["divergences"]["confirmed"]:
                confidence += weights["divergences"]

            # Evaluar momentum
            if abs(indicators["macd"]["momentum_strength"]) > 0.7:
                confidence += weights["momentum"]
            elif abs(indicators["macd"]["momentum_strength"]) > 0.5:
                confidence += weights["momentum"] * 0.7

            return min(confidence, 1.0)

        except Exception as e:
            print(f"Error calculando confianza técnica: {str(e)}")
            return 0.0

    def _validate_time_continuity(self, timestamps: List[int]) -> bool:
        """Valida la continuidad temporal de los datos"""
        try:
            expected_interval = timestamps[1] - timestamps[0]
            max_allowed_gap = expected_interval * 2

            for i in range(1, len(timestamps)):
                if timestamps[i] - timestamps[i - 1] > max_allowed_gap:
                    return False
            return True
        except Exception:
            return False

    def _validate_price_coherence(
        self, highs: List[float], lows: List[float], closes: List[float]
    ) -> bool:
        """Valida la coherencia de los precios"""
        try:
            for i in range(len(highs)):
                if not (lows[i] <= closes[i] <= highs[i]):
                    return False
                if highs[i] < lows[i]:
                    return False
            return True
        except Exception:
            return False

    def analyze_trading_opportunity(self, symbol: str) -> Optional[TradeRecommendation]:
        try:
            ticker_data = self.client.get_ticker_24h(symbol)
            if not ticker_data:
                return None

            current_price = float(ticker_data["lastPrice"])
            market_data = {
                "D1": self.client.get_klines(symbol, "1d", 200),
                "H4": self.client.get_klines(symbol, "4h", 200),
                "H1": self.client.get_klines(symbol, "1h", 200),
            }

            # Obtener análisis básico para niveles y momentum
            trend = self._analyze_main_trend(market_data)
            momentum = self._analyze_momentum(market_data["H4"])
            volatility = self._calculate_volatility(market_data["H4"])
            timing = self.analyze_entry_timing(symbol)

            # Tomar decisión basada principalmente en timing
            if timing.timing == EntryTiming.IMMEDIATE and timing.confidence >= 0.7:
                if volatility <= self.strategy_params["volatility"]["max"]:
                    signal = TradingSignal.BUY
                    strength = SignalStrength.STRONG
                    reasons = [
                        "Momento óptimo para entrada inmediata",
                        f"Precio cerca del soporte",
                        f"Volatilidad controlada ({volatility:.1%})",
                        f"Alta confianza en timing ({timing.confidence:.1%})",
                    ]
                    if trend["trend"] == "bullish":
                        reasons.append(f"Tendencia alcista apoya la entrada")
                    if momentum["is_positive"]:
                        reasons.append("Momentum positivo confirma entrada")
                else:
                    signal = TradingSignal.HOLD
                    strength = SignalStrength.WEAK
                    reasons = [f"Volatilidad excesiva ({volatility:.1%})"]

            elif (
                timing.timing == EntryTiming.WAIT_BREAKOUT and timing.confidence >= 0.6
            ):
                signal = TradingSignal.HOLD
                strength = SignalStrength.MODERATE
                reasons = [
                    "Esperando confirmación de ruptura",
                    f"Confianza en timing: {timing.confidence:.1%}",
                ]
                if trend["trend"] == "bullish":
                    reasons.append(
                        f"Tendencia alcista con fuerza {trend['strength']:.2%}"
                    )

            elif timing.timing == EntryTiming.WAIT_DIP:
                signal = TradingSignal.HOLD
                strength = SignalStrength.MODERATE
                reasons = ["Esperando retroceso para mejor entrada"]

            else:
                signal = TradingSignal.HOLD
                strength = SignalStrength.WEAK
                reasons = ["Condiciones de mercado no óptimas"]

            # Calcular niveles de entrada/salida
            levels = self._calculate_dynamic_levels(
                current_price, trend, momentum, volatility, timing
            )

            return TradeRecommendation(
                signal=signal,
                strength=strength,
                reasons=reasons,
                entry_price=levels["entry"],
                stop_loss=levels["stop_loss"],
                take_profit=levels["take_profit"],
            )

        except Exception as e:
            print(f"Error en analyze_trading_opportunity: {str(e)}")
            return None

    def _calculate_dynamic_levels(
        self,
        current_price: float,
        trend: Dict,
        momentum: Dict,
        volatility: float,
        timing: TimingWindow,
    ) -> Dict:
        """Calcula niveles dinámicos de trading basados en múltiples factores"""
        try:
            # Determinar multiplicadores basados en condiciones
            momentum_strength = self._get_momentum_strength(momentum)
            volatility_level = self._get_volatility_level(volatility)

            # Ajustar multiplicadores
            tp_multiplier = (
                self.take_profit_params["momentum_multiplier"][momentum_strength]
                * self.take_profit_params["volatility_adjustment"][volatility_level]
            )

            # Calcular entrada base
            base_entry = current_price
            if timing.timing == EntryTiming.WAIT_DIP:
                base_entry = current_price * 0.99  # Esperar -1%
            elif timing.timing == EntryTiming.WAIT_BREAKOUT:
                base_entry = current_price * 1.01  # Esperar +1%

            # Calcular stop loss dinámico
            stop_distance = self._calculate_dynamic_stop(
                current_price, volatility, trend
            )
            stop_loss = base_entry - stop_distance

            # Calcular take profits dinámicos
            take_profits = []
            accumulated_size = 0
            for level in self.take_profit_params["base_levels"]:
                target = base_entry * (1 + (level["level"] - 1) * tp_multiplier)
                take_profits.append(
                    {
                        "price": target,
                        "size": level["size"],
                        "accumulated": accumulated_size + level["size"],
                    }
                )
                accumulated_size += level["size"]

            return {
                "entry": base_entry,
                "stop_loss": stop_loss,
                "take_profit": take_profits[-1]["price"],  # Último nivel
                "take_profit_levels": take_profits,
                "trading_params": {
                    "momentum_strength": momentum_strength,
                    "volatility_level": volatility_level,
                    "tp_multiplier": tp_multiplier,
                },
            }

        except Exception as e:
            print(f"Error en calculate_dynamic_levels: {str(e)}")
            return {
                "entry": current_price,
                "stop_loss": current_price * 0.98,
                "take_profit": current_price * 1.04,
            }

    def _evaluate_volume_score(self, levels: Dict) -> float:
        """Evalúa la puntuación del volumen y liquidez"""
        try:
            score = 0.0
            weight = 0.0

            # 1. Volumen relativo al promedio
            if "volume" in levels:
                volume_ratio = levels["volume"].get("ratio", 0)
                if volume_ratio > 2.0:  # Volumen muy significativo
                    score += 0.4
                elif volume_ratio > 1.5:  # Volumen significativo
                    score += 0.3
                weight += 0.4

            # 2. Presión compradora
            if "volume" in levels and "buy_pressure" in levels["volume"]:
                buy_pressure = levels["volume"]["buy_pressure"]
                if buy_pressure > 0.7:  # Alta presión compradora
                    score += 0.3
                elif buy_pressure > 0.6:
                    score += 0.2
                weight += 0.3

            # 3. Consistencia del volumen
            if "volume" in levels and "is_increasing" in levels["volume"]:
                if levels["volume"]["is_increasing"]:
                    score += 0.3
                weight += 0.3

            return score / weight if weight > 0 else 0.0

        except Exception as e:
            print(f"Error en evaluate_volume_score: {str(e)}")
            return 0.0

    def _determine_final_signal(
        self, trend: Dict, momentum: Dict, timing: TimingWindow, levels: Dict
    ) -> Tuple[TradingSignal, SignalStrength]:
        try:
            score = 0
            confirmations = 0
            total_required_confirmations = 3  # Reducido de 5
            confidence_threshold = 0.6  # Reducido de 0.75

            # Evaluación de Tendencia (35%)
            trend_score = self._evaluate_trend_score(trend)
            score += trend_score * 0.35
            if trend_score > 0.6:  # Reducido de 0.7
                confirmations += 1

            # Evaluación de Momentum (25%)
            momentum_score = self._evaluate_momentum_score(momentum)
            score += momentum_score * 0.25
            if momentum_score > 0.6:  # Reducido de 0.7
                confirmations += 1

            # Evaluación de Timing (25%)
            timing_score = self._evaluate_timing_score(timing, levels)
            score += timing_score * 0.25
            if timing_score > 0.6:  # Reducido de 0.7
                confirmations += 1

            # Evaluación de Volumen (15%)
            volume_score = self._evaluate_volume_score(levels)
            score += volume_score * 0.15
            if volume_score > 0.6:
                confirmations += 1

            # Determinar señal
            if score >= 0.70 and confirmations >= total_required_confirmations:
                return TradingSignal.BUY, SignalStrength.STRONG
            elif score >= 0.60 and confirmations >= (total_required_confirmations - 1):
                return TradingSignal.BUY, SignalStrength.MODERATE
            elif score <= 0.30 and confirmations >= total_required_confirmations:
                return TradingSignal.SELL, SignalStrength.STRONG
            elif score <= 0.40 and confirmations >= (total_required_confirmations - 1):
                return TradingSignal.SELL, SignalStrength.MODERATE

            return TradingSignal.HOLD, SignalStrength.WEAK

        except Exception as e:
            print(ConsoleColors.error(f"Error en determine_final_signal: {str(e)}"))
            return TradingSignal.HOLD, SignalStrength.WEAK

    def _evaluate_trend_score(self, trend: Dict) -> float:
        """Evalúa la puntuación de tendencia con múltiples timeframes"""
        try:
            score = 0.0
            weight = 0.0

            # 1. Alineación de EMAs
            if trend.get("timeframes"):
                for tf_data in trend["timeframes"].values():
                    if tf_data.get("is_bullish"):
                        ema_alignment = (
                            tf_data["price"]
                            > tf_data["ema20"]
                            > tf_data["ema50"]
                            > tf_data["ema200"]
                        )
                        if ema_alignment:
                            score += 0.4
                            weight += 0.4

            # 2. Fuerza de tendencia
            strength = trend.get("strength", 0)
            score += strength * 0.3
            weight += 0.3

            # 3. Momentum de tendencia
            if trend.get("timeframes"):
                momentum_count = sum(
                    1
                    for tf in trend["timeframes"].values()
                    if tf.get("momentum", 0) > 0
                )
                momentum_score = momentum_count / len(trend["timeframes"])
                score += momentum_score * 0.3
                weight += 0.3

            return score / weight if weight > 0 else 0.0

        except Exception as e:
            print(f"Error en evaluate_trend_score: {str(e)}")
            return 0.0

    def _evaluate_momentum_score(self, momentum: Dict) -> float:
        """Evalúa la puntuación de momentum con múltiples factores"""
        try:
            score = 0.0
            weight = 0.0

            # 1. Momentum a corto plazo
            if "short_term" in momentum:
                short_term = momentum["short_term"]
                if short_term > 0:
                    score += min(short_term / 10, 1.0) * 0.4
                weight += 0.4

            # 2. Momentum a medio plazo
            if "medium_term" in momentum:
                medium_term = momentum["medium_term"]
                if medium_term > 0:
                    score += min(medium_term / 20, 1.0) * 0.3
                weight += 0.3

            # 3. Aceleración
            if "acceleration" in momentum:
                acceleration = momentum["acceleration"]
                if acceleration > 0:
                    score += min(acceleration / 5, 1.0) * 0.3
                weight += 0.3

            # 4. Consistencia
            is_strong = momentum.get("is_strong", False)
            is_positive = momentum.get("is_positive", False)
            if is_strong and is_positive:
                score += 0.2
                weight += 0.2

            return score / weight if weight > 0 else 0.0

        except Exception as e:
            print(f"Error en evaluate_momentum_score: {str(e)}")
            return 0.0

    def _evaluate_timing_score(self, timing: TimingWindow, levels: Dict) -> float:
        """Evalúa la puntuación de timing considerando múltiples factores"""
        try:
            score = 0.0
            weight = 0.0

            # 1. Timing base
            if timing.timing == EntryTiming.IMMEDIATE:
                score += timing.confidence * 0.4
                weight += 0.4
            elif timing.timing == EntryTiming.WAIT_BREAKOUT:
                # Solo puntuar si estamos cerca de niveles clave
                if (
                    levels.get("entry")
                    and abs(levels["entry"] - timing.target_price) / timing.target_price
                    < 0.02
                ):
                    score += timing.confidence * 0.3
                weight += 0.3

            # 2. Proximidad a niveles clave
            if levels.get("support") and levels.get("entry"):
                distance_to_support = (
                    abs(levels["entry"] - levels["support"]) / levels["support"]
                )
                if distance_to_support < 0.02:  # 2% del soporte
                    score += 0.3
                    weight += 0.3

            # 3. Risk/Reward
            if (
                levels.get("stop_loss")
                and levels.get("take_profit")
                and levels.get("entry")
            ):
                risk = abs(levels["entry"] - levels["stop_loss"])
                reward = abs(levels["take_profit"] - levels["entry"])
                if risk > 0:
                    rr_ratio = reward / risk
                    if rr_ratio >= 3:
                        score += 0.3
                        weight += 0.3

            return score / weight if weight > 0 else 0.0

        except Exception as e:
            print(f"Error en evaluate_timing_score: {str(e)}")
            return 0.0

    def _validate_signal_conditions(
        self, trend: Dict, momentum: Dict, timing: TimingWindow, levels: Dict
    ) -> bool:
        """Validaciones adicionales de seguridad para la señal"""
        try:
            # 1. Verificar consistencia de tendencia
            if not trend.get("is_valid"):
                return False

            # 2. Verificar momentum mínimo
            if not momentum.get("is_positive"):
                return False

            # 3. Verificar timing válido
            if timing.confidence < 0.5:
                return False

            # 4. Verificar niveles válidos
            if (
                not levels.get("entry")
                or not levels.get("stop_loss")
                or not levels.get("take_profit")
            ):
                return False

            # 5. Verificar risk/reward mínimo
            risk = abs(levels["entry"] - levels["stop_loss"])
            reward = abs(levels["take_profit"] - levels["entry"])
            if risk == 0 or (reward / risk) < 2:
                return False

            return True

        except Exception as e:
            print(f"Error en validate_signal_conditions: {str(e)}")
            return False

    def _calculate_dynamic_stop(
        self, current_price: float, volatility: float, trend: Dict
    ) -> float:
        """Calcula el stop loss dinámico basado en volatilidad y tendencia"""
        try:
            # Base stop distance (2% por defecto)
            base_stop = current_price * 0.02

            # Ajustar por volatilidad
            if volatility > 0.05:  # Alta volatilidad
                base_stop *= 1.5
            elif volatility < 0.02:  # Baja volatilidad
                base_stop *= 0.8

            # Ajustar por fuerza de tendencia
            trend_strength = trend.get("strength", 0.5)
            if trend_strength > 0.7:
                base_stop *= 0.9  # Reducir stop en tendencia fuerte
            elif trend_strength < 0.3:
                base_stop *= 1.2  # Aumentar stop en tendencia débil

            return base_stop

        except Exception as e:
            print(f"Error en calculate_dynamic_stop: {str(e)}")
            return current_price * 0.02

    def _get_momentum_strength(self, momentum: Dict) -> str:
        """Determina la fuerza del momentum"""
        try:
            momentum_value = momentum.get("medium_term", 0)
            if abs(momentum_value) > 15:
                return "strong"
            elif abs(momentum_value) > 8:
                return "moderate"
            return "weak"
        except Exception:
            return "weak"

    def _get_volatility_level(self, volatility: float) -> str:
        """Determina el nivel de volatilidad"""
        if volatility > 0.05:
            return "high"
        elif volatility < 0.02:
            return "low"
        return "normal"

    def _generate_comprehensive_reasons(
        self, trend: Dict, momentum: Dict, timing: TimingWindow, levels: Dict
    ) -> List[str]:
        """Genera razones detalladas para la recomendación"""
        reasons = []

        # Razones de tendencia
        if trend["trend"] == "bullish":
            reasons.append(f"Tendencia alcista con fuerza {trend['strength']:.2%}")
        elif trend["trend"] == "bearish":
            reasons.append(f"Tendencia bajista con fuerza {trend['strength']:.2%}")

        # Razones de momentum
        momentum_str = self._get_momentum_strength(momentum)
        if momentum_str == "strong":
            reasons.append("Momentum fuerte y favorable")
        elif momentum_str == "moderate":
            reasons.append("Momentum moderado")

        # Razones de timing
        if timing.timing != EntryTiming.NOT_RECOMMENDED:
            reasons.append(f"Timing favorable: {timing.timing.value}")
            reasons.append(f"Confianza en timing: {timing.confidence:.1%}")

        return reasons

    def _calculate_market_score(self, market_metrics: Dict) -> float:
        """Calcula score basado en métricas de mercado"""
        try:
            score = 0.0
            weight = 0.0

            # Evaluar cambio de precio
            if "percent_change_24h" in market_metrics:
                change_24h = market_metrics["percent_change_24h"]
                score += ((change_24h + 20) / 40) * 0.3  # Normalizar a [0,1]
                weight += 0.3

            # Evaluar volumen
            if "volume_24h" in market_metrics:
                volume = market_metrics["volume_24h"]
                volume_score = min(volume / 100000000, 1.0)  # Normalizar a 100M
                score += volume_score * 0.3
                weight += 0.3

            # Evaluar dominancia de mercado
            if "market_dominance" in market_metrics:
                dominance = market_metrics["market_dominance"]
                score += (dominance / 100) * 0.2
                weight += 0.2

            # Evaluar ranking
            if "rank" in market_metrics:
                rank_score = 1 - (min(market_metrics["rank"], 100) / 100)
                score += rank_score * 0.2
                weight += 0.2

            return score / weight if weight > 0 else 0.5

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando market score: {str(e)}"))
            return 0.5

    def _generate_analysis_reasons(
        self,
        trend_analysis: Dict,
        technical_analysis: Dict,
        volume_analysis: Dict,
        support_resistance: Dict,
        volatility: float,
        market_metrics: Dict,
    ) -> List[str]:
        """Genera razones de análisis incluyendo métricas de mercado"""
        reasons = []

        # Razones técnicas existentes
        if trend_analysis.get("trend") == "bullish":
            reasons.append(
                f"Tendencia alcista con fuerza {trend_analysis.get('strength', 0):.2%}"
            )
        elif trend_analysis.get("trend") == "bearish":
            reasons.append(
                f"Tendencia bajista con fuerza {trend_analysis.get('strength', 0):.2%}"
            )

        # Razones de mercado
        if market_metrics:
            change_24h = market_metrics.get("percent_change_24h", 0)
            if abs(change_24h) > 5:
                direction = "alcista" if change_24h > 0 else "bajista"
                reasons.append(
                    f"Movimiento {direction} significativo ({change_24h:.1f}%)"
                )

            volume_24h = market_metrics.get("volume_24h", 0)
            if volume_24h > 100000000:  # 100M USD
                reasons.append(f"Alto volumen de mercado (${volume_24h:,.0f})")

            rank = market_metrics.get("rank", 0)
            if rank <= 20:
                reasons.append(f"Top {rank} por capitalización de mercado")

            dominance = market_metrics.get("market_dominance", 0)
            if dominance > 1:
                reasons.append(
                    f"Dominancia de mercado significativa ({dominance:.1f}%)"
                )

        # Mantener el resto de las razones existentes...

        return reasons

    def generate_trading_levels(
        self, current_price: float, volatility: float, market_metrics: Dict = None
    ) -> Dict:
        """Genera niveles de trading considerando métricas de mercado"""
        try:
            # Ajustar los niveles según la volatilidad del mercado
            base_stop_loss = 0.02  # 2% base
            base_take_profit = 0.06  # 6% base

            # Ajustar según métricas de mercado si están disponibles
            if market_metrics:
                market_volatility = market_metrics.get("volatility_7d", volatility)
                rank = market_metrics.get("rank", 50)

                # Ajustar según ranking
                if rank <= 10:
                    base_stop_loss *= 0.8  # Menor riesgo para top coins
                    base_take_profit *= 0.8
                elif rank > 50:
                    base_stop_loss *= 1.2  # Mayor riesgo para coins menores
                    base_take_profit *= 1.2

                # Ajustar según volatilidad del mercado
                volatility_factor = max(0.5, min(2.0, market_volatility / 0.02))
                base_stop_loss *= volatility_factor
                base_take_profit *= volatility_factor

            # Calcular niveles
            stop_loss = current_price * (1 - base_stop_loss)
            take_profits = [
                {
                    "price": current_price * (1 + (base_take_profit * level)),
                    "size": size,
                }
                for level, size in [
                    (1.0, 0.4),  # 40% en primer objetivo
                    (1.5, 0.3),  # 30% en segundo objetivo
                    (2.0, 0.3),  # 30% en tercer objetivo
                ]
            ]

            return {
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "take_profits": take_profits,
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error generando niveles: {str(e)}"))
            return None

    def _determine_trading_signal_with_market_data(
        self,
        trend_analysis: Dict,
        technical_analysis: Dict,
        volume_analysis: Dict,
        support_resistance: Dict,
        current_price: float,
        volatility: float,
        market_metrics: Dict,
    ) -> Tuple[TradingSignal, SignalStrength]:
        """Determina señal considerando datos técnicos y de mercado"""
        try:
            # Calcular scores individuales
            technical_score = self._calculate_technical_score(
                trend_analysis, technical_analysis
            )
            market_score = self._calculate_market_score(market_metrics)
            volume_score = self._calculate_volume_score(volume_analysis)
            momentum_score = self._calculate_momentum_score(technical_analysis)

            # Calcular score final ponderado
            final_score = (
                technical_score * self.analysis_weights["technical"]
                + market_score * self.analysis_weights["market"]
                + volume_score * self.analysis_weights["volume"]
                + momentum_score * self.analysis_weights["momentum"]
            )

            # Determinar señal y fuerza
            if final_score >= 0.7:
                return TradingSignal.BUY, SignalStrength.STRONG
            elif final_score >= 0.5:
                return TradingSignal.BUY, SignalStrength.MODERATE
            elif final_score <= 0.3:
                return TradingSignal.SELL, SignalStrength.STRONG
            elif final_score <= 0.5:
                return TradingSignal.SELL, SignalStrength.MODERATE
            else:
                return TradingSignal.HOLD, SignalStrength.WEAK

        except Exception as e:
            print(ConsoleColors.error(f"Error determinando señal: {str(e)}"))
            return TradingSignal.HOLD, SignalStrength.WEAK

    def format_trading_levels(self, recommendation: TradeRecommendation) -> str:
        """Formatea los niveles de trading para la salida"""
        output = []

        if not recommendation.entry_price:
            return "No hay niveles disponibles"

        output.append(f"Niveles de Precio:")
        output.append(f"Precio actual: ${recommendation.entry_price:,.6f}")

        if recommendation.stop_loss:
            stop_loss_percent = (
                (recommendation.stop_loss - recommendation.entry_price)
                / recommendation.entry_price
            ) * 100
            output.append(
                f"Stop Loss: ${recommendation.stop_loss:.8f} ({stop_loss_percent:.2f}%)"
            )

        if recommendation.take_profits:
            output.append("\nTake Profits Escalonados:")
            for idx, tp in enumerate(recommendation.take_profits, 1):
                profit_percent = (
                    (tp["price"] - recommendation.entry_price)
                    / recommendation.entry_price
                ) * 100
                output.append(
                    f"{idx}. ${tp['price']:.8f} (+{profit_percent:.2f}%) - "
                    f"Vender {tp['size'] * 100}% de la posición"
                )

        return "\n".join(output)

    def validate_trading_levels(self, levels: Dict) -> bool:
        """Valida que los niveles de trading sean coherentes"""
        try:
            if not levels or "entry_price" not in levels:
                return False

            # Validar stop loss
            if levels["stop_loss"] >= levels["entry_price"]:
                return False

            # Validar take profits
            last_price = levels["entry_price"]
            for tp in levels["take_profits"]:
                if tp["price"] <= last_price:
                    return False
                last_price = tp["price"]

            return True
        except Exception as e:
            print(f"Error validando niveles: {e}")
            return False

    def manage_position(
        self, symbol: str, entry_price: float, current_positions: List[Dict]
    ):
        """Gestiona posiciones abiertas"""
        try:
            current_price = float(self.client.get_ticker_price(symbol)["price"])
            profit_percentage = ((current_price - entry_price) / entry_price) * 100

            # Mover a breakeven (+0.5%)
            if profit_percentage >= 0.5:
                new_stop = entry_price * 1.001  # Entry + 0.1%
                self.update_stop_loss(symbol, new_stop)

            # Activar trailing stop
            if profit_percentage >= 2:  # Activar en +2%
                trail_amount = current_price * 0.005  # 0.5% trailing
                new_stop = current_price - trail_amount
                self.update_stop_loss(symbol, new_stop)

            # Gestionar take profits parciales
            for position in current_positions:
                self.check_take_profit_levels(symbol, position, current_price)

        except Exception as e:
            print(f"Error gestionando posición: {e}")

    def check_take_profit_levels(
        self, symbol: str, position: Dict, current_price: float
    ):
        """Verifica y ejecuta take profits parciales"""
        try:
            entry_price = float(position["entry_price"])
            current_profit = (current_price - entry_price) / entry_price * 100

            for level in self.strategy_params["risk_params"]["take_profit_levels"]:
                target_profit = (
                    level["ratio"] - 1
                ) * 100  # Convertir ratio a porcentaje
                if current_profit >= target_profit:
                    quantity = float(position["quantity"]) * level["size"]
                    if quantity > 0:
                        self.execute_take_profit(symbol, quantity, current_price)

        except Exception as e:
            print(f"Error verificando take profits: {e}")

    def execute_take_profit(self, symbol: str, quantity: float, price: float):
        """Ejecuta una orden de take profit"""
        try:
            # Aquí implementarías la lógica de tu broker/exchange
            print(f"Ejecutando take profit en {symbol}: {quantity} a ${price:,.2f}")
            # self.client.create_market_sell_order(symbol, quantity)
        except Exception as e:
            print(f"Error ejecutando take profit: {e}")

    def _determine_trading_signal(
        self,
        trend_analysis,
        technical_analysis,
        volume_analysis,
        support_resistance,
        current_price,
        volatility,
    ) -> Tuple[TradingSignal, SignalStrength]:
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
                trend = trend_analysis.get("trend")
                strength = trend_analysis.get("strength", 0)

                if trend == "bullish":
                    trend_score = 40 * strength
                    score += trend_score
                    if strength > 0.6:
                        confirmations += 2
                    elif strength > 0.3:
                        confirmations += 1
                    conditions_met.append(f"Tendencia alcista ({strength:.1%})")
                elif trend == "bearish":
                    trend_score = -40 * strength
                    score += trend_score
                    if strength > 0.6:
                        confirmations += 2
                    elif strength > 0.3:
                        confirmations += 1
                    conditions_met.append(f"Tendencia bajista ({strength:.1%})")

            # 3. Análisis de Soporte/Resistencia (25 puntos máximo)
            if support_resistance:
                support = support_resistance.get("support", 0)
                resistance = support_resistance.get("resistance", 0)

                if support and resistance:
                    distance_to_support = (current_price - support) / support
                    distance_to_resistance = (
                        resistance - current_price
                    ) / current_price

                    # Cerca del soporte (señal de compra)
                    if distance_to_support <= 0.02:  # 2% del soporte
                        score += 25
                        confirmations += 1
                        conditions_met.append("Precio cerca del soporte")
                    # Cerca de la resistencia (señal de venta)
                    elif distance_to_resistance <= 0.02:  # 2% de la resistencia
                        score -= 25
                        confirmations += 1
                        conditions_met.append("Precio cerca de la resistencia")

            # 4. Análisis de Volumen (20 puntos máximo)
            if volume_analysis:
                volume_ratio = volume_analysis.get("ratio", 1.0)
                is_increasing = volume_analysis.get("is_increasing", False)

                if volume_ratio > 2.0:  # Volumen significativo
                    if is_increasing:
                        score += 20
                        confirmations += 1
                        conditions_met.append("Volumen creciente significativo")
                    else:
                        score += 10
                elif volume_ratio < 0.5:  # Volumen bajo
                    score -= 10
                    conditions_met.append("Volumen bajo")

            # 5. Análisis Técnico (15 puntos máximo)
            if technical_analysis:
                # RSI
                rsi = technical_analysis.get("rsi", 50)
                if rsi <= 30:  # Sobreventa
                    score += 15
                    confirmations += 1
                    conditions_met.append(f"RSI en sobreventa ({rsi:.1f})")
                elif rsi >= 70:  # Sobrecompra
                    score -= 15
                    confirmations += 1
                    conditions_met.append(f"RSI en sobrecompra ({rsi:.1f})")

                # MACD
                macd = technical_analysis.get("macd", {})
                if macd.get("crossover") == "bullish":
                    score += 10
                    confirmations += 1
                    conditions_met.append("Cruce alcista de MACD")
                elif macd.get("crossover") == "bearish":
                    score -= 10
                    confirmations += 1
                    conditions_met.append("Cruce bajista de MACD")

            # Determinar señal y fuerza basado en score y confirmaciones
            if (
                score >= self.signal_thresholds["score"]["strong_buy"]
                and confirmations >= self.signal_thresholds["confirmations"]["strong"]
            ):
                return TradingSignal.BUY, SignalStrength.STRONG
            elif (
                score >= self.signal_thresholds["score"]["buy"]
                and confirmations >= self.signal_thresholds["confirmations"]["moderate"]
            ):
                return TradingSignal.BUY, SignalStrength.MODERATE
            elif (
                score <= self.signal_thresholds["score"]["strong_sell"]
                and confirmations >= self.signal_thresholds["confirmations"]["strong"]
            ):
                return TradingSignal.SELL, SignalStrength.STRONG
            elif (
                score <= self.signal_thresholds["score"]["sell"]
                and confirmations >= self.signal_thresholds["confirmations"]["moderate"]
            ):
                return TradingSignal.SELL, SignalStrength.MODERATE
            else:
                strength = (
                    SignalStrength.MODERATE if abs(score) > 30 else SignalStrength.WEAK
                )
                return TradingSignal.HOLD, strength

        except Exception as e:
            print(ConsoleColors.error(f"Error en determine_trading_signal: {str(e)}"))
            return TradingSignal.HOLD, SignalStrength.WEAK

    def _generate_hold_recommendation(
        self, reason: str, current_price: Optional[float] = None
    ) -> TradeRecommendation:
        """Genera una recomendación HOLD con manejo mejorado de precios nulos"""
        if current_price is None:
            return TradeRecommendation(
                signal=TradingSignal.HOLD,
                strength=SignalStrength.WEAK,
                reasons=[reason],
                entry_price=None,
                stop_loss=None,
                take_profit=None,
            )

        stop_loss = current_price * 0.95  # -5%
        take_profit = current_price * 1.15  # +15%

        return TradeRecommendation(
            signal=TradingSignal.HOLD,
            strength=SignalStrength.WEAK,
            reasons=[reason],
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    def _analyze_main_trend(self, market_data):
        """
        Analiza la tendencia principal con los datos proporcionados.
        """
        try:
            trend_analysis = {
                "is_valid": False,
                "trend": "neutral",
                "strength": 0,
                "data": {},
                "timeframes": {},
            }

            for timeframe, data in market_data.items():
                if not data or len(data) < 50:  # Mínimo de velas necesarias
                    continue

                # Extraer valores de cierre numéricos
                closes = []
                try:
                    for candle in data:
                        if isinstance(candle, dict):
                            close = float(candle.get("close", 0))
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
                    "price": closes[-1],
                    "ema20": ema20,
                    "ema50": ema50,
                    "ema200": ema200,
                    "momentum": momentum,
                    "is_bullish": closes[-1] > ema20 > ema50 > ema200,
                    "is_bearish": closes[-1] < ema20 < ema50 < ema200,
                }

                trend_analysis["timeframes"][timeframe] = timeframe_trend
                trend_analysis["data"][timeframe] = closes

            # Analizar tendencia global
            timeframe_count = len(trend_analysis["timeframes"])
            if timeframe_count > 0:
                bullish_count = sum(
                    1
                    for tf in trend_analysis["timeframes"].values()
                    if tf["is_bullish"]
                )
                bearish_count = sum(
                    1
                    for tf in trend_analysis["timeframes"].values()
                    if tf["is_bearish"]
                )

                trend_analysis["is_valid"] = True
                if bullish_count > timeframe_count / 2:
                    trend_analysis["trend"] = "bullish"
                    trend_analysis["strength"] = bullish_count / timeframe_count
                elif bearish_count > timeframe_count / 2:
                    trend_analysis["trend"] = "bearish"
                    trend_analysis["strength"] = bearish_count / timeframe_count
                else:
                    trend_analysis["trend"] = "neutral"
                    trend_analysis["strength"] = 0.5

            return trend_analysis

        except Exception as e:
            print(f"Error en análisis de tendencia principal: {str(e)}")
            return {
                "is_valid": False,
                "trend": "neutral",
                "strength": 0,
                "data": {},
                "timeframes": {},
            }

    # Mantener métodos existentes
    def _analyze_trend(self, candlesticks: List[Dict]) -> MarketTrend:
        """Analiza la tendencia del mercado"""
        try:
            closes = [float(candle["close"]) for candle in candlesticks]

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
                    "ratio": 1.0,
                    "is_significant": False,
                    "is_increasing": False,
                    "average": 0,
                    "buy_pressure": 0,
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
                    "ratio": 1.0,
                    "is_significant": False,
                    "is_increasing": False,
                    "average": 0,
                    "buy_pressure": 0,
                }

            # Cálculos de volumen
            avg_volume = sum(volumes[-20:]) / 20
            current_volume = volumes[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

            # Calcular presión compradora
            recent_volumes = volumes[-5:]
            recent_price_changes = [
                closes[i] - closes[i - 1] for i in range(len(closes) - 5, len(closes))
            ]

            buy_volume = sum(
                vol
                for vol, change in zip(recent_volumes, recent_price_changes)
                if change > 0
            )
            total_volume = sum(recent_volumes)
            buy_pressure = buy_volume / total_volume if total_volume > 0 else 0.5

            return {
                "ratio": volume_ratio,
                "is_significant": volume_ratio
                > self.thresholds["volume"]["significant"],
                "is_increasing": sum(volumes[-3:]) > sum(volumes[-6:-3]),
                "average": avg_volume,
                "buy_pressure": buy_pressure,
            }

        except Exception as e:
            print(f"Error en análisis de volumen: {e}")
            print(f"Tipo de candlesticks: {type(candlesticks)}")
            if candlesticks and len(candlesticks) > 0:
                print(f"Tipo de primera vela: {type(candlesticks[0])}")
                print(f"Contenido de primera vela: {candlesticks[0]}")
            return {
                "ratio": 1.0,
                "is_significant": False,
                "is_increasing": False,
                "average": 0,
                "buy_pressure": 0,
            }

    def _analyze_momentum(self, candlesticks: List[Dict]) -> Dict:
        """Analiza el momentum del precio"""
        try:
            closes = [float(candle["close"]) for candle in candlesticks]

            if len(closes) < 42:
                return {
                    "short_term": 0,
                    "medium_term": 0,
                    "is_strong": False,
                    "is_positive": False,
                    "acceleration": 0,
                }

            # Calcular cambios porcentuales
            change_24h = ((closes[-1] - closes[-6]) / closes[-6]) * 100
            change_7d = ((closes[-1] - closes[-42]) / closes[-42]) * 100

            # Calcular aceleración del momentum
            recent_changes = [
                ((closes[i] - closes[i - 1]) / closes[i - 1]) * 100
                for i in range(-5, 0)
            ]
            acceleration = sum(recent_changes) / len(recent_changes)

            return {
                "short_term": change_24h,
                "medium_term": change_7d,
                "is_strong": abs(change_24h)
                > self.thresholds["momentum"]["strong_buy"],
                "is_positive": change_24h > 0 and change_7d > 0,
                "acceleration": acceleration,
            }

        except Exception:
            return {
                "short_term": 0,
                "medium_term": 0,
                "is_strong": False,
                "is_positive": False,
                "acceleration": 0,
            }

    # Mantener otros métodos existentes y agregar nuevos
    def _analyze_multiple_timeframes(self, market_data: Dict) -> Dict:
        mtf_analysis = {}

        for timeframe, data in market_data.items():
            try:
                closes = [float(candle["close"]) for candle in data]
                volumes = [float(candle["volume"]) for candle in data]

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
                    self.indicators["macd"]["signal"],
                )

                mtf_analysis[timeframe] = {
                    "emas": emas,
                    "rsis": rsis,
                    "macd": macd,
                    "volume_trend": self._analyze_volume_trend(volumes),
                    "price_trend": self._analyze_price_trend(closes),
                }

            except Exception as e:
                print(
                    ConsoleColors.warning(f"Error en análisis de {timeframe}: {str(e)}")
                )
                continue

        return mtf_analysis

    def _analyze_price_patterns(self, candlesticks: List[Dict]) -> Dict:
        """Identifica patrones de precio avanzados"""
        try:
            closes = [float(candle["close"]) for candle in candlesticks]
            highs = [float(candle["high"]) for candle in candlesticks]
            lows = [float(candle["low"]) for candle in candlesticks]
            opens = [float(candle["open"]) for candle in candlesticks]

            patterns = []

            # Doble fondo
            if (
                lows[-3] > lows[-2]
                and lows[-2] < lows[-1]
                and abs(lows[-3] - lows[-1]) < (highs[-2] - lows[-2]) * 0.1
            ):
                patterns.append(
                    {"type": "bullish", "name": "Doble Fondo", "reliability": 0.8}
                )

            # Doble techo
            if (
                highs[-3] < highs[-2]
                and highs[-2] > highs[-1]
                and abs(highs[-3] - highs[-1]) < (highs[-2] - lows[-2]) * 0.1
            ):
                patterns.append(
                    {"type": "bearish", "name": "Doble Techo", "reliability": 0.8}
                )

            # Martillo alcista
            for i in range(-3, 0):
                body = abs(opens[i] - closes[i])
                shadow_lower = min(opens[i], closes[i]) - lows[i]
                shadow_upper = highs[i] - max(opens[i], closes[i])
                if (
                    shadow_lower > body * 2
                    and shadow_upper < body * 0.5
                    and closes[i] > opens[i]
                ):
                    patterns.append(
                        {
                            "type": "bullish",
                            "name": "Martillo Alcista",
                            "reliability": 0.7,
                        }
                    )

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
                            patterns.append(
                                {
                                    "type": "bearish",
                                    "name": "Pin Bar Bajista",
                                    "reliability": 0.75,
                                }
                            )
                        if lower_wick > 2 * body and upper_wick < 0.2 * total_length:
                            patterns.append(
                                {
                                    "type": "bullish",
                                    "name": "Pin Bar Alcista",
                                    "reliability": 0.75,
                                }
                            )

            return {
                "patterns": patterns,
                "pattern_count": len(patterns),
                "dominant_bias": (
                    "bullish"
                    if sum(1 for p in patterns if p["type"] == "bullish")
                    > sum(1 for p in patterns if p["type"] == "bearish")
                    else "bearish"
                ),
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error en análisis de patrones: {str(e)}"))
            return {"patterns": [], "pattern_count": 0, "dominant_bias": "neutral"}

    def _analyze_market_correlation(self, symbol: str) -> Dict:
        """Analiza la correlación con BTC y el mercado general"""
        try:
            # Obtener datos de BTC y el par analizado
            btc_data = self.client.get_klines("BTCUSDT", interval="1h", limit=168)
            symbol_data = self.client.get_klines(symbol, interval="1h", limit=168)

            # Calcular correlación
            btc_returns = self._calculate_returns(
                [float(candle["close"]) for candle in btc_data]
            )
            symbol_returns = self._calculate_returns(
                [float(candle["close"]) for candle in symbol_data]
            )

            correlation = np.corrcoef(btc_returns, symbol_returns)[0, 1]

            # Calcular beta (volatilidad relativa a BTC)
            beta = np.std(symbol_returns) / np.std(btc_returns)

            # Analizar fuerza de mercado
            market_strength = self._analyze_market_strength()

            return {
                "correlation": correlation,
                "beta": beta,
                "btc_influence": (
                    "high"
                    if abs(correlation) > 0.7
                    else "moderate" if abs(correlation) > 0.4 else "low"
                ),
                "market_strength": market_strength,
                "risk_level": (
                    "high" if beta > 1.5 else "moderate" if beta > 1 else "low"
                ),
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error en análisis de correlación: {str(e)}"))
            return {
                "correlation": 0,
                "beta": 1,
                "btc_influence": "unknown",
                "market_strength": "neutral",
                "risk_level": "moderate",
            }

    def _analyze_market_strength(self) -> str:
        """Analiza la fuerza general del mercado"""
        try:
            # Obtener datos de BTC
            btc_data = self.client.get_klines("BTCUSDT", interval="1d", limit=14)
            if not btc_data:
                return "neutral"

            closes = [float(candle["close"]) for candle in btc_data]
            volumes = [float(candle["volume"]) for candle in btc_data]

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
            print(
                ConsoleColors.error(f"Error en análisis de fuerza de mercado: {str(e)}")
            )
            return "neutral"

    def _combine_all_analysis(
        self,
        trend: MarketTrend,
        volume_analysis: Dict,
        momentum: Dict,
        rsi: float,
        mtf_analysis: Dict,
        pattern_analysis: Dict,
        correlation: Dict,
    ) -> Dict:
        """Combina todos los análisis para generar un análisis final"""
        try:
            # Calcular score base
            base_score = 0
            confidence = 0
            signals = []

            # Análisis de tendencia (35%)
            if trend in [MarketTrend.STRONG_UPTREND, MarketTrend.UPTREND]:
                base_score += 35
                signals.append("Tendencia alcista")
                confidence += 0.2
            elif trend in [MarketTrend.STRONG_DOWNTREND, MarketTrend.DOWNTREND]:
                base_score -= 35
                signals.append("Tendencia bajista")
                confidence += 0.2

            # Análisis de volumen (25%)
            if volume_analysis["is_significant"] and volume_analysis["is_increasing"]:
                base_score += 25
                signals.append("Volumen significativo y creciente")
                confidence += 0.15

            # Análisis de momentum (20%)
            if momentum["is_strong"] and momentum["is_positive"]:
                base_score += 20
                signals.append("Momentum fuerte y positivo")
                confidence += 0.15
            elif momentum["is_strong"] and not momentum["is_positive"]:
                base_score -= 20
                signals.append("Momentum fuerte pero negativo")
                confidence += 0.15

            # Análisis de patrones y correlación (20%)
            if pattern_analysis:
                for pattern in pattern_analysis.get("patterns", []):
                    if pattern["type"] == "bullish":
                        base_score += 10 * pattern.get("strength", 0.5)
                        signals.append(f"Patrón alcista: {pattern['name']}")
                        confidence += 0.1
                    elif pattern["type"] == "bearish":
                        base_score -= 10 * pattern.get("strength", 0.5)
                        signals.append(f"Patrón bajista: {pattern['name']}")
                        confidence += 0.1

            if correlation.get("market_strength") == "very_strong":
                base_score *= 1.2
                confidence += 0.1
            elif correlation.get("market_strength") == "weak":
                base_score *= 0.8
                confidence += 0.1

            return {
                "final_score": base_score,
                "confidence": min(confidence, 1.0),
                "signals": signals,
                "market_context": {
                    "trend": trend.value if hasattr(trend, "value") else str(trend),
                    "volume_context": volume_analysis,
                    "momentum_context": momentum,
                    "correlation_context": correlation,
                },
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error combinando análisis: {str(e)}"))
            return {
                "final_score": 0,
                "confidence": 0,
                "signals": ["Error en análisis"],
                "market_context": {},
            }

    def _calculate_returns(self, prices: List[float]) -> List[float]:
        """Calcula los retornos porcentuales"""
        return [
            (prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))
        ]

    def _calculate_bollinger_bands(
        self, prices: List[float], period: int = 20, std_dev: float = 2.0
    ) -> Dict:
        """Calcula las bandas de Bollinger"""
        try:
            if len(prices) < period:
                return {"upper": prices[-1], "middle": prices[-1], "lower": prices[-1]}

            # Calcular SMA y desviación estándar
            sma = sum(prices[-period:]) / period
            std = np.std(prices[-period:])

            return {
                "upper": sma + (std_dev * std),
                "middle": sma,
                "lower": sma - (std_dev * std),
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando Bollinger Bands: {str(e)}"))
            return {"upper": prices[-1], "middle": prices[-1], "lower": prices[-1]}

    def _calculate_macd(
        self,
        prices: List[float],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Dict:
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
                "macd": macd_line[-1],
                "signal": signal_line[-1],
                "histogram": histogram[-1],
                "trending_up": macd_line[-1] > signal_line[-1],
                "momentum_strength": abs(histogram[-1]),
                "crossover": (
                    "bullish"
                    if macd_line[-1] > signal_line[-1]
                    and macd_line[-2] <= signal_line[-2]
                    else (
                        "bearish"
                        if macd_line[-1] < signal_line[-1]
                        and macd_line[-2] >= signal_line[-2]
                        else "none"
                    )
                ),
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando MACD: {str(e)}"))
            return {
                "macd": 0,
                "signal": 0,
                "histogram": 0,
                "trending_up": False,
                "momentum_strength": 0,
                "crossover": "none",
            }

    def _calculate_advanced_macd(self, closes: List[float]) -> Dict:
        """
        Calcula MACD con análisis avanzado
        """
        try:
            fast = 12
            slow = 26
            signal = 9

            # Calcular EMAs
            fast_ema = self._calculate_ema(closes, fast)
            slow_ema = self._calculate_ema(closes, slow)

            # Calcular línea MACD
            macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]

            # Calcular línea de señal
            signal_line = self._calculate_ema(macd_line, signal)

            # Calcular histograma
            histogram = [m - s for m, s in zip(macd_line, signal_line)]

            # Analizar tendencia y fuerza
            trend = "bullish" if macd_line[-1] > signal_line[-1] else "bearish"
            strength = (
                abs(histogram[-1]) / abs(macd_line[-1]) if abs(macd_line[-1]) > 0 else 0
            )

            # Detectar convergencias/divergencias
            price_trend = "up" if closes[-1] > closes[-5] else "down"
            macd_trend = "up" if macd_line[-1] > macd_line[-5] else "down"
            divergence = price_trend != macd_trend

            return {
                "macd": float(macd_line[-1]),
                "signal": float(signal_line[-1]),
                "histogram": float(histogram[-1]),
                "trending_up": trend == "bullish",
                "strength": float(strength),
                "momentum_strength": float(abs(histogram[-1])),
                "divergence": divergence,
                "crossover": (
                    "bullish"
                    if macd_line[-1] > signal_line[-1]
                    and macd_line[-2] <= signal_line[-2]
                    else (
                        "bearish"
                        if macd_line[-1] < signal_line[-1]
                        and macd_line[-2] >= signal_line[-2]
                        else "none"
                    )
                ),
            }

        except Exception as e:
            print(f"Error en _calculate_advanced_macd: {str(e)}")
            return {
                "macd": 0,
                "signal": 0,
                "histogram": 0,
                "trending_up": False,
                "strength": 0,
                "momentum_strength": 0,
                "divergence": False,
                "crossover": "none",
            }

    def _calculate_advanced_bollinger(self, timeframes: Dict) -> Dict:
        """
        Calcula Bandas de Bollinger con análisis avanzado
        """
        try:
            results = {}
            for tf_name, tf_info in timeframes.items():
                prices = tf_info["data"]
                if len(prices) < 20:
                    continue

                # Calcular media móvil y desviación estándar
                sma = sum(prices[-20:]) / 20
                std = np.std(prices[-20:])

                # Calcular bandas
                upper = sma + (2 * std)
                lower = sma - (2 * std)

                # Calcular ancho de banda y %B
                bandwidth = (upper - lower) / sma
                percent_b = (
                    (prices[-1] - lower) / (upper - lower) if upper != lower else 0.5
                )

                results[tf_name] = {
                    "upper": float(upper),
                    "middle": float(sma),
                    "lower": float(lower),
                    "bandwidth": float(bandwidth),
                    "percent_b": float(percent_b),
                    "squeeze": bandwidth < 0.1,  # Detectar squeeze
                }

            # Calcular señales combinadas
            combined = {
                "squeeze_momentum": any(tf["squeeze"] for tf in results.values()),
                "trend_strength": self._calculate_bb_trend_strength(results),
                "volatility_increasing": self._is_volatility_increasing(results),
            }

            return {**results, "analysis": combined}

        except Exception as e:
            print(f"Error en _calculate_advanced_bollinger: {str(e)}")
            return {"error": str(e), "is_valid": False}

    def _calculate_bb_trend_strength(self, bb_results: Dict) -> float:
        """
        Calcula la fuerza de la tendencia basada en Bollinger Bands
        """
        try:
            strength = 0.0
            count = 0

            for tf_data in bb_results.values():
                if isinstance(tf_data, dict):
                    percent_b = tf_data.get("percent_b", 0.5)
                    if percent_b > 0.8:
                        strength += 1
                    elif percent_b < 0.2:
                        strength -= 1
                    count += 1

            return strength / count if count > 0 else 0

        except Exception:
            return 0.0

    def _is_volatility_increasing(self, bb_results: Dict) -> bool:
        """
        Determina si la volatilidad está aumentando
        """
        try:
            bandwidths = [
                tf_data.get("bandwidth", 0)
                for tf_data in bb_results.values()
                if isinstance(tf_data, dict)
            ]

            if len(bandwidths) >= 2:
                return bandwidths[-1] > bandwidths[-2]
            return False

        except Exception:
            return False

    def _validate_indicator_consistency(self, indicators: Dict) -> Dict:
        """
        Valida la consistencia entre diferentes indicadores
        """
        try:
            consistency = {
                "rsi_macd_aligned": False,
                "trend_momentum_aligned": False,
                "volume_confirms": False,
                "score": 0.0,
            }

            # Verificar alineación RSI-MACD
            rsi_bullish = indicators["rsi"].get("weighted_value", 50) < 70
            macd_bullish = indicators["macd"].get("trending_up", False)
            consistency["rsi_macd_aligned"] = rsi_bullish == macd_bullish

            # Verificar alineación tendencia-momentum
            trend_bullish = indicators["trend"].get("direction", "") == "bullish"
            momentum_bullish = indicators["macd"].get("momentum_strength", 0) > 0
            consistency["trend_momentum_aligned"] = trend_bullish == momentum_bullish

            # Calcular score final
            score = 0.0
            if consistency["rsi_macd_aligned"]:
                score += 0.5
            if consistency["trend_momentum_aligned"]:
                score += 0.5

            consistency["score"] = score

            return consistency

        except Exception as e:
            print(f"Error en _validate_indicator_consistency: {str(e)}")
            return {"score": 0.0}

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calcula el RSI con manejo mejorado de errores y tipos de datos"""
        try:
            # Validación y conversión de entrada
            if isinstance(prices[0], dict):
                prices = [float(price["close"]) for price in prices]
            elif isinstance(prices[0], (list, tuple)):
                prices = [float(price[4]) for price in prices]
            else:
                prices = [float(price) for price in prices]

            if len(prices) < period + 1:
                return 50.0

            # Calcular cambios
            deltas = []
            for i in range(1, len(prices)):
                deltas.append(prices[i] - prices[i - 1])

            # Separar ganancias y pérdidas
            gains = []
            losses = []
            for delta in deltas:
                if delta > 0:
                    gains.append(delta)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(delta))

            # Calcular promedios
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period

            if avg_loss == 0:
                return 100.0
            if avg_gain == 0:
                return 0.0

            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

            return float(max(0.0, min(100.0, rsi)))

        except Exception as e:
            print(f"Error en _calculate_rsi: {str(e)}")
            return 50.0

    def analyze_entry_timing(self, symbol: str) -> TimingWindow:
        """Analiza el mejor momento para entrar al mercado"""
        try:
            # Obtener datos
            candlesticks = self.client.get_klines(symbol, interval="1h", limit=200)
            current_price = float(self.client.get_ticker_price(symbol)["price"])

            if not candlesticks:
                return TimingWindow(
                    timing=EntryTiming.NOT_RECOMMENDED,
                    timeframe="N/A",
                    conditions=["No hay suficientes datos históricos"],
                )

            # Análisis técnico
            technical = self._analyze_technical_indicators(candlesticks)

            # Validar resultado del análisis técnico
            if not technical.get("is_valid", False):
                return TimingWindow(
                    timing=EntryTiming.NOT_RECOMMENDED,
                    timeframe="N/A",
                    conditions=["Error en análisis técnico"],
                )

            support_resistance = self._calculate_support_resistance(candlesticks)
            volatility = self._calculate_volatility(candlesticks)

            # Inicializar variables
            timing = EntryTiming.NOT_RECOMMENDED
            timeframe = "N/A"
            target_price = current_price
            confidence = 0.0
            conditions = []

            # Verificar RSI
            rsi_value = technical.get("rsi", 50.0)
            if isinstance(rsi_value, dict):
                rsi_value = rsi_value.get("value", 50.0)

            if rsi_value <= 30:
                timing = EntryTiming.IMMEDIATE
                timeframe = "0-4 horas"
                confidence = 0.8
                conditions.append(f"RSI en sobreventa ({rsi_value:.1f})")
            elif rsi_value >= 70:
                timing = EntryTiming.WAIT_DIP
                timeframe = "12-24 horas"
                confidence = 0.6
                conditions.append(f"RSI en sobrecompra ({rsi_value:.1f})")

            # Verificar niveles de soporte/resistencia
            if support_resistance:
                support = support_resistance.get("support")
                resistance = support_resistance.get("resistance")

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
                        conditions.append(
                            f"Precio cerca de resistencia (${resistance:,.2f})"
                        )
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
                conditions=conditions,
            )

        except Exception as e:
            print(f"Error en determine_timing_window: {str(e)}")
            return TimingWindow(
                timing=EntryTiming.NOT_RECOMMENDED,
                timeframe="N/A",
                conditions=["Error en análisis"],
            )

    def _analyze_technical_indicators(self, candlesticks: List[Dict]) -> Dict:
        """Analiza los indicadores técnicos con mejor manejo de errores"""
        try:
            if not candlesticks or len(candlesticks) < 50:
                return {
                    "is_valid": False,
                    "rsi": 50.0,
                    "macd": {"trend": "neutral", "histogram": 0, "crossover": "none"},
                    "bollinger": {"upper": 0, "middle": 0, "lower": 0},
                }

            # Extraer y validar datos
            closes = []
            for candle in candlesticks:
                try:
                    if isinstance(candle, (list, tuple)):
                        closes.append(float(candle[4]))
                    elif isinstance(candle, dict):
                        closes.append(float(candle["close"]))
                except (IndexError, KeyError, ValueError) as e:
                    print(f"Error procesando vela: {e}")
                    continue

            if len(closes) < 50:
                return {
                    "is_valid": False,
                    "rsi": 50.0,
                    "macd": {"trend": "neutral", "histogram": 0, "crossover": "none"},
                    "bollinger": {"upper": 0, "middle": 0, "lower": 0},
                }

            # Calcular indicadores
            rsi = self._calculate_rsi(closes[-14:])
            macd = self._calculate_macd(closes)
            bb = self._calculate_bollinger_bands(closes[-20:])

            return {
                "is_valid": True,
                "rsi": float(rsi),
                "macd": macd,
                "bollinger": bb,
                "last_price": closes[-1],
            }

        except Exception as e:
            print(f"Error en análisis técnico: {str(e)}")
            return {
                "is_valid": False,
                "rsi": 50.0,
                "macd": {"trend": "neutral", "histogram": 0, "crossover": "none"},
                "bollinger": {"upper": 0, "middle": 0, "lower": 0},
            }

    def _calculate_volatility(
        self, candlesticks: List[Dict], period: int = 14
    ) -> float:
        """Calcula la volatilidad usando True Range"""
        try:
            if not candlesticks or len(candlesticks) < period:
                return 0.0

            # Extraer precios
            highs = [float(candle["high"]) for candle in candlesticks]
            lows = [float(candle["low"]) for candle in candlesticks]
            closes = [float(candle["close"]) for candle in candlesticks]

            # Calcular True Ranges
            true_ranges = []
            for i in range(1, len(candlesticks)):
                true_range = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]),
                )
                true_ranges.append(true_range)

            # Calcular ATR
            atr = sum(true_ranges[-period:]) / period if true_ranges else 0

            # Calcular cambios porcentuales
            daily_returns = [
                (closes[i] - closes[i - 1]) / closes[i - 1]
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

    def _calculate_support_resistance(self, candlesticks: List[Dict]) -> Dict:
        """Calcula niveles de soporte y resistencia"""
        try:
            closes = [float(candle["close"]) for candle in candlesticks]
            highs = [float(candle["high"]) for candle in candlesticks]
            lows = [float(candle["low"]) for candle in candlesticks]

            # Usar pivots para identificar niveles
            window = 20
            supports = []
            resistances = []

            for i in range(window, len(candlesticks) - window):
                # Identificar pivots
                if min(lows[i - window : i + window]) == lows[i]:
                    supports.append(lows[i])
                if max(highs[i - window : i + window]) == highs[i]:
                    resistances.append(highs[i])

            # Calcular niveles finales
            support = max(supports[-3:]) if supports else min(lows)
            resistance = min(resistances[-3:]) if resistances else max(highs)

            return {
                "support": support,
                "resistance": resistance,
                "support_levels": sorted(supports[-3:]),
                "resistance_levels": sorted(resistances[-3:], reverse=True),
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando S/R: {str(e)}"))
            return {"support": 0, "resistance": 0}

    def _analyze_volume_trend(self, volumes: List[float]) -> Dict:
        """Analiza la tendencia del volumen"""
        try:
            if len(volumes) < 20:
                return {"trend": "neutral", "strength": 0}

            short_ma = sum(volumes[-5:]) / 5
            long_ma = sum(volumes[-20:]) / 20

            trend = "increasing" if short_ma > long_ma else "decreasing"
            strength = abs((short_ma - long_ma) / long_ma)

            return {
                "trend": trend,
                "strength": strength,
                "short_ma": short_ma,
                "long_ma": long_ma,
            }

        except Exception as e:
            print(
                ConsoleColors.error(
                    f"Error en análisis de tendencia de volumen: {str(e)}"
                )
            )
            return {"trend": "neutral", "strength": 0}

    def _analyze_price_trend(self, closes: List[float]) -> Dict:
        """Analiza la tendencia del precio"""
        try:
            if len(closes) < 20:
                return {"trend": "neutral", "strength": 0}

            ema_short = self._calculate_ema(closes, 7)
            ema_long = self._calculate_ema(closes, 21)

            trend = "bullish" if ema_short[-1] > ema_long[-1] else "bearish"
            strength = abs((ema_short[-1] - ema_long[-1]) / ema_long[-1])

            return {
                "trend": trend,
                "strength": strength,
                "ema_short": ema_short[-1],
                "ema_long": ema_long[-1],
            }

        except Exception as e:
            print(
                ConsoleColors.error(
                    f"Error en análisis de tendencia de precio: {str(e)}"
                )
            )
            return {"trend": "neutral", "strength": 0}

    def _identify_pattern(self, candlesticks: List[Dict]) -> Dict:
        """Identifica patrones de velas"""
        try:
            opens = [float(candle["open"]) for candle in candlesticks[-5:]]
            closes = [float(candle["close"]) for candle in candlesticks[-5:]]
            highs = [float(candle["high"]) for candle in candlesticks[-5:]]
            lows = [float(candle["low"]) for candle in candlesticks[-5:]]

            patterns = []

            # Doji
            if abs(opens[-1] - closes[-1]) <= (highs[-1] - lows[-1]) * 0.1:
                patterns.append({"name": "Doji", "type": "neutral"})

            # Martillo
            body = abs(opens[-1] - closes[-1])
            lower_wick = min(opens[-1], closes[-1]) - lows[-1]
            upper_wick = highs[-1] - max(opens[-1], closes[-1])

            if lower_wick > body * 2 and upper_wick < body * 0.5:
                patterns.append({"name": "Hammer", "type": "bullish"})

            # Estrella fugaz
            if upper_wick > body * 2 and lower_wick < body * 0.5:
                patterns.append({"name": "Shooting Star", "type": "bearish"})

            return {
                "patterns": patterns,
                "count": len(patterns),
                "bias": patterns[0]["type"] if patterns else "neutral",
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error identificando patrones: {str(e)}"))
            return {"patterns": [], "count": 0, "bias": "neutral"}

    def _calculate_trade_levels(
        self,
        trend_analysis: Dict,
        technical_analysis: Dict,
        market_conditions: Dict,
        current_price: float,
    ) -> Dict:
        """Calcula niveles de entrada y salida precisos"""
        try:
            # Calcular ATR para stops dinámicos
            atr = self._calculate_atr(market_conditions["data"])

            # Encontrar niveles clave
            support_resistance = self._calculate_support_resistance(
                market_conditions["data"]
            )

            # Calcular entrada
            if trend_analysis["trend"] == "bullish":
                entry = max(
                    current_price,  # No entrar arriba del precio actual
                    support_resistance["support"] * 1.01,  # 1% sobre soporte
                )

                # Stop loss basado en ATR y soporte
                stop_loss = max(
                    entry - (atr * 2),
                    support_resistance["support"] * 0.99,  # 1% bajo soporte
                )

                # Take profit en dos niveles
                tp1 = entry + (
                    (entry - stop_loss)
                    * self.strategy_params["risk_params"]["partial_tp_ratio"]
                )
                tp2 = min(
                    entry + (atr * 4),
                    support_resistance["resistance"] * 0.99,  # 1% bajo resistencia
                )
            else:
                return None

            return {
                "entry": entry,
                "stop_loss": stop_loss,
                "take_profit_1": tp1,
                "take_profit_2": tp2,
                "risk_reward_ratio": (tp2 - entry) / (entry - stop_loss),
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
                high = float(candlesticks[i]["high"])
                low = float(candlesticks[i]["low"])
                prev_close = float(candlesticks[i - 1]["close"])

                tr = max(
                    [
                        high - low,  # Rango actual
                        abs(
                            high - prev_close
                        ),  # Movimiento desde el cierre anterior al máximo
                        abs(
                            low - prev_close
                        ),  # Movimiento desde el cierre anterior al mínimo
                    ]
                )
                true_ranges.append(tr)

            return sum(true_ranges[-period:]) / period

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando ATR: {str(e)}"))
            return 0.0

    def _adjust_timeframe(self, timeframe: str, multiplier: float) -> str:
        """Ajusta el timeframe según el multiplicador"""
        try:
            import re

            numbers = re.findall(r"\d+", timeframe)
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
            if trend["trend"] != "bullish":
                return {
                    "is_valid": False,
                    "optimal_entry": current_price,
                    "reason": "Tendencia no favorable",
                }

            # Obtener niveles de soporte y resistencia
            support = levels.get("support", current_price * 0.95)
            resistance = levels.get("resistance", current_price * 1.05)

            # Calcular entrada óptima basada en niveles
            if current_price < support * 1.02:  # Cerca del soporte
                optimal_entry = support * 1.01
            elif current_price > resistance * 0.98:  # Cerca de la resistencia
                optimal_entry = None  # No entrar cerca de resistencia
            else:
                optimal_entry = current_price

            # Validar con volumen
            if volume.get("is_significant", False) and volume.get(
                "is_increasing", False
            ):
                confidence = "high"
            else:
                confidence = "low"

            # Validar con patrones
            if patterns.get("pattern_type") == "bullish":
                confidence = "high"
            elif patterns.get("pattern_type") == "bearish":
                optimal_entry = None

            if optimal_entry is None:
                return {
                    "is_valid": False,
                    "optimal_entry": current_price,
                    "reason": "No se encontró punto de entrada favorable",
                }

            return {
                "is_valid": True,
                "optimal_entry": optimal_entry,
                "reason": f"Entrada óptima encontrada con confianza {confidence}",
                "support": support,
                "resistance": resistance,
            }

        except Exception as e:
            print(f"Error calculando entrada óptima: {str(e)}")
            return {
                "is_valid": False,
                "optimal_entry": current_price,
                "reason": "Error en cálculo",
            }

    def _validate_volume(self, candlesticks: List[Dict]) -> bool:
        """
        Valida que el volumen sea suficiente y consistente
        """
        try:
            volumes = [float(candle["volume"]) for candle in candlesticks[-20:]]
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

    def _generate_entry_reason(
        self, conditions: Dict, trend: MarketTrend, rsi: float, volume_analysis: Dict
    ) -> List[str]:
        """Genera razones detalladas para la entrada"""
        reasons = []

        if conditions["trend_alignment"]:
            reasons.append("Alineación alcista de EMAs (8>21>50>200)")

        if conditions["rsi_condition"]:
            reasons.append(f"RSI en zona óptima ({rsi:.1f})")

        if conditions["volume_condition"]:
            reasons.append(
                f"Volumen significativo ({volume_analysis['ratio']:.1f}x promedio)"
            )

        if conditions["volatility_condition"]:
            reasons.append("Volatilidad controlada en rango óptimo")

        if conditions["price_position"]:
            reasons.append("Precio por encima de EMA 200 (tendencia principal alcista)")

        if trend == MarketTrend.STRONG_UPTREND:
            reasons.append("Fuerte tendencia alcista confirmada")

        return reasons

    def _analyze_candle_patterns(self, candlesticks: List[Dict]) -> Dict:
        """Analiza patrones de velas japonesas"""
        try:
            opens = [float(candle["open"]) for candle in candlesticks[-5:]]
            closes = [float(candle["close"]) for candle in candlesticks[-5:]]
            highs = [float(candle["high"]) for candle in candlesticks[-5:]]
            lows = [float(candle["low"]) for candle in candlesticks[-5:]]

            patterns = []

            # Patrón envolvente alcista
            if (
                closes[-2] < opens[-2]
                and closes[-1] > opens[-1]
                and opens[-1] < closes[-2]
                and closes[-1] > opens[-2]
            ):
                patterns.append(
                    {"name": "Bullish Engulfing", "type": "bullish", "strength": 0.8}
                )

            # Patrón envolvente bajista
            if (
                closes[-2] > opens[-2]
                and closes[-1] < opens[-1]
                and opens[-1] > closes[-2]
                and closes[-1] < opens[-2]
            ):
                patterns.append(
                    {"name": "Bearish Engulfing", "type": "bearish", "strength": 0.8}
                )

            # Martillo alcista
            body = abs(opens[-1] - closes[-1])
            lower_shadow = min(opens[-1], closes[-1]) - lows[-1]
            upper_shadow = highs[-1] - max(opens[-1], closes[-1])

            if lower_shadow > (2 * body) and upper_shadow < (0.5 * body):
                patterns.append({"name": "Hammer", "type": "bullish", "strength": 0.7})

            # Otros patrones relevantes...

            return {
                "has_confirmation": len(patterns) > 0,
                "patterns": patterns,
                "pattern_name": patterns[0]["name"] if patterns else "No Pattern",
                "pattern_type": patterns[0]["type"] if patterns else "neutral",
                "pattern_strength": patterns[0]["strength"] if patterns else 0,
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando patrones de velas: {str(e)}"))
            return {
                "has_confirmation": False,
                "patterns": [],
                "pattern_name": "Error",
                "pattern_type": "neutral",
                "pattern_strength": 0,
            }

    def _generate_market_condition_reason(
        self, volatility: float, volume_analysis: Dict, momentum: Dict
    ) -> str:
        """Genera razón detallada de las condiciones de mercado"""
        reasons = []

        if volatility > self.strategy_params["risk_params"]["volatility_threshold"]:
            reasons.append(f"Volatilidad muy alta ({volatility:.1%})")

        if volume_analysis["ratio"] < self.thresholds["volume"]["significant"]:
            reasons.append(
                f"Volumen insuficiente ({volume_analysis['ratio']:.1f}x promedio)"
            )

        if momentum["is_strong"] and not momentum["is_positive"]:
            reasons.append("Momentum fuertemente negativo")

        return " | ".join(reasons) if reasons else "Condiciones de mercado favorables"

    def _determine_signal_strength(
        self, trend_analysis, volume_analysis, pattern_analysis, entry_analysis
    ):
        """
        Determina la señal y su fuerza basándose en múltiples factores
        """
        try:
            score = 0
            conditions_met = 0
            total_conditions = 4

            # 1. Análisis de tendencia (0-40 puntos)
            if trend_analysis.get("trend") == "bullish":
                score += 40 * trend_analysis.get("strength", 0)
                conditions_met += 1
            elif trend_analysis.get("trend") == "bearish":
                score -= 40 * trend_analysis.get("strength", 0)

            # 2. Análisis de volumen (0-30 puntos)
            if volume_analysis.get("is_significant", False):
                score += 15
                conditions_met += 1
            if volume_analysis.get("is_increasing", False):
                score += 15
                conditions_met += 1

            # 3. Patrones de precio (0-20 puntos)
            if pattern_analysis and pattern_analysis.get("patterns"):
                if pattern_analysis.get("pattern_type") == "bullish":
                    score += 20 * pattern_analysis.get("pattern_strength", 0)
                    conditions_met += 1
                elif pattern_analysis.get("pattern_type") == "bearish":
                    score -= 20 * pattern_analysis.get("pattern_strength", 0)

            # 4. Entrada óptima (0-10 puntos)
            if entry_analysis and entry_analysis.get("is_valid", False):
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

    def _generate_specific_reasons(
        self, trend_analysis, volume_analysis, pattern_analysis, entry_analysis
    ):
        """
        Genera razones específicas basadas en el análisis
        """
        try:
            reasons = []

            # Razones de tendencia
            if trend_analysis.get("trend") == "bullish":
                reasons.append(
                    f"Tendencia alcista con fuerza {trend_analysis.get('strength', 0):.2%}"
                )
            elif trend_analysis.get("trend") == "bearish":
                reasons.append(
                    f"Tendencia bajista con fuerza {trend_analysis.get('strength', 0):.2%}"
                )

            # Razones de volumen
            if volume_analysis.get("is_significant"):
                reasons.append(
                    f"Volumen significativo ({volume_analysis.get('ratio', 0):.1f}x promedio)"
                )
            if volume_analysis.get("is_increasing"):
                reasons.append("Volumen creciente")

            # Razones de patrones
            if pattern_analysis and pattern_analysis.get("patterns"):
                reasons.append(
                    f"Patrón {pattern_analysis.get('pattern_name', 'desconocido')} detectado"
                )

            # Razones de entrada
            if entry_analysis.get("is_valid"):
                reasons.append(
                    f"Punto de entrada óptimo identificado en ${entry_analysis.get('optimal_entry', 0):.8f}"
                )
                if entry_analysis.get("reason"):
                    reasons.append(entry_analysis["reason"])
            else:
                reasons.append("No se encontró punto de entrada óptimo")

            return reasons if reasons else ["Análisis incompleto o sin señales claras"]

        except Exception as e:
            print(f"Error generando razones: {str(e)}")
            return ["Error en análisis"]

    def calculate_position_sizes(self, capital: float, current_price: float) -> Dict:
        """Calcula los tamaños de posición para cada nivel"""
        try:
            risk_amount = capital * (
                self.strategy_params["risk_params"]["max_risk_percent"] / 100
            )
            position_sizes = []

            for tp_level in self.strategy_params["risk_params"]["take_profit_levels"]:
                size = risk_amount * tp_level["size"]
                position_sizes.append(
                    {
                        "size": size / current_price,
                        "take_profit": current_price * tp_level["ratio"],
                    }
                )

            return {
                "position_sizes": position_sizes,
                "total_size": sum(p["size"] for p in position_sizes),
            }
        except Exception as e:
            print(f"Error calculando tamaños de posición: {e}")
            return None

    def calculate_exit_levels(self, entry_price: float, stop_loss: float) -> Dict:
        """Calcula los niveles de salida basados en el riesgo"""
        try:
            risk = entry_price - stop_loss
            levels = []

            for tp_level in self.strategy_params["risk_params"]["take_profit_levels"]:
                take_profit = entry_price + (risk * tp_level["ratio"])
                levels.append(
                    {
                        "price": take_profit,
                        "size": tp_level["size"],
                        "ratio": tp_level["ratio"],
                    }
                )

            return {
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit_levels": levels,
                "breakeven_price": entry_price
                + (
                    risk * self.strategy_params["risk_params"]["stop_loss"]["breakeven"]
                ),
                "trailing_activation": entry_price
                + (
                    risk
                    * self.strategy_params["risk_params"]["stop_loss"]["trailing"][
                        "activation"
                    ]
                ),
            }
        except Exception as e:
            print(f"Error calculando niveles de salida: {e}")
            return None

    def update_trailing_stop(
        self,
        current_price: float,
        high_price: float,
        entry_price: float,
        current_stop: float,
    ) -> float:
        """Actualiza el trailing stop dinámicamente"""
        try:
            # Calcular distancia inicial
            initial_risk = entry_price - current_stop
            activation_level = entry_price + (
                initial_risk
                * self.strategy_params["risk_params"]["stop_loss"]["trailing"][
                    "activation"
                ]
            )

            # Si el precio no ha alcanzado el nivel de activación, mantener stop actual
            if high_price < activation_level:
                return current_stop

            # Calcular nuevo trailing stop
            trailing_distance = (
                current_price
                * self.strategy_params["risk_params"]["stop_loss"]["trailing"]["step"]
            )
            new_stop = current_price - trailing_distance

            # Retornar el mayor entre el nuevo stop y el actual
            return max(new_stop, current_stop)

        except Exception as e:
            print(f"Error actualizando trailing stop: {e}")
            return current_stop

    def should_take_profit(
        self,
        current_price: float,
        entry_price: float,
        position_size: float,
        profits_taken: List[float],
    ) -> Optional[float]:
        """Determina si se debe tomar beneficios y qué cantidad"""
        try:
            current_profit = (current_price - entry_price) / entry_price

            for i, tp_level in enumerate(
                self.strategy_params["risk_params"]["take_profit_levels"]
            ):
                # Verificar si este nivel ya fue tomado
                if i < len(profits_taken):
                    continue

                target_profit = tp_level["ratio"]
                if current_profit >= target_profit:
                    return position_size * tp_level["size"]

            return None

        except Exception as e:
            print(f"Error evaluando toma de beneficios: {e}")
            return None

    def should_move_stop_loss(
        self,
        current_price: float,
        entry_price: float,
        current_stop: float,
        high_price: float,
    ) -> Optional[float]:
        """Determina si se debe mover el stop loss y a qué nivel"""
        try:
            profit_percent = (current_price - entry_price) / entry_price
            initial_risk = entry_price - current_stop

            # Mover a breakeven
            breakeven_target = self.strategy_params["risk_params"]["stop_loss"][
                "breakeven"
            ]
            if profit_percent >= breakeven_target and current_stop < entry_price:
                return entry_price + (
                    initial_risk * 0.1
                )  # Breakeven + 10% del riesgo inicial

            # Actualizar trailing stop
            if (
                profit_percent
                >= self.strategy_params["risk_params"]["stop_loss"]["trailing"][
                    "activation"
                ]
            ):
                return self.update_trailing_stop(
                    current_price, high_price, entry_price, current_stop
                )

            return current_stop

        except Exception as e:
            print(f"Error evaluando stop loss: {e}")
            return current_stop

    def validate_exit_conditions(
        self, current_price: float, entry_price: float, technical_indicators: Dict
    ) -> Tuple[bool, str]:
        """Valida condiciones adicionales de salida"""
        try:
            # Obtener indicadores
            rsi = technical_indicators.get("rsi", 50)
            volume_ratio = technical_indicators.get("volume_ratio", 1.0)
            trend_strength = technical_indicators.get("trend_strength", 0.5)

            # Verificar condiciones de salida
            if (
                rsi
                >= self.strategy_params["thresholds"]["exit_conditions"][
                    "rsi_overbought"
                ]
            ):
                return True, "RSI en sobrecompra"

            if (
                trend_strength
                < self.strategy_params["thresholds"]["exit_conditions"][
                    "trend_reversal"
                ]
            ):
                return True, "Debilitamiento de tendencia"

            if (
                volume_ratio
                < self.strategy_params["thresholds"]["exit_conditions"]["volume_drop"]
            ):
                return True, "Caída significativa de volumen"

            return False, ""

        except Exception as e:
            print(f"Error validando condiciones de salida: {e}")
            return False, "Error en validación"

    def _calculate_technical_score(
        self, trend_analysis: Dict, technical_analysis: Dict
    ) -> float:
        """Calcula score basado en análisis técnico"""
        try:
            score = 0.0
            weight = 0.0

            # Evaluar tendencia
            if trend_analysis.get("trend"):
                trend_strength = trend_analysis.get("strength", 0)
                if trend_analysis["trend"] == "bullish":
                    score += trend_strength * 0.4
                elif trend_analysis["trend"] == "bearish":
                    score -= trend_strength * 0.4
                weight += 0.4

            # Evaluar RSI
            rsi = technical_analysis.get("rsi", 50)
            if rsi <= 30:  # Sobreventa
                score += 0.3
            elif rsi >= 70:  # Sobrecompra
                score -= 0.3
            weight += 0.3

            # Evaluar MACD
            macd = technical_analysis.get("macd", {})
            if macd.get("crossover") == "bullish":
                score += 0.3
            elif macd.get("crossover") == "bearish":
                score -= 0.3
            weight += 0.3

            return (score / weight + 1) / 2 if weight > 0 else 0.5

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando technical score: {str(e)}"))
            return 0.5

    def _calculate_volume_score(self, volume_analysis: Dict) -> float:
        """Calcula score basado en análisis de volumen"""
        try:
            score = 0.0
            weight = 0.0

            # Evaluar ratio de volumen
            if "ratio" in volume_analysis:
                volume_score = min(volume_analysis["ratio"] / 2, 1.0)
                score += volume_score * 0.6
                weight += 0.6

            # Evaluar tendencia de volumen
            if volume_analysis.get("is_increasing"):
                score += 0.4
                weight += 0.4

            return score / weight if weight > 0 else 0.5

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando volume score: {str(e)}"))
            return 0.5

    def _calculate_momentum_score(self, technical_analysis: Dict) -> float:
        """Calcula score basado en momentum"""
        try:
            score = 0.0
            weight = 0.0

            # Evaluar RSI momentum
            rsi = technical_analysis.get("rsi", 50)
            rsi_score = (rsi - 50) / 50  # Normalizar a [-1, 1]
            score += (rsi_score + 1) / 2 * 0.4  # Convertir a [0, 1]
            weight += 0.4

            # Evaluar MACD momentum
            macd = technical_analysis.get("macd", {})
            if macd.get("trending_up"):
                score += 0.3
            weight += 0.3

            # Evaluar Bollinger Bands
            bb = technical_analysis.get("bollinger", {})
            if bb:
                price = technical_analysis.get("last_price", 0)
                upper = bb.get("upper", price)
                lower = bb.get("lower", price)

                if price > upper:
                    score += 0.3
                elif price < lower:
                    score -= 0.3
                weight += 0.3

            return max(0, min(1, score / weight)) if weight > 0 else 0.5

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando momentum score: {str(e)}"))
            return 0.5

    # def analyze_new_listings(self, days: int = 7, limit: int = 10) -> List[Dict]:
    #     """Analiza nuevas criptomonedas en el mercado de forma dinámica"""
    #     try:
    #         # Obtener nuevos listings
    #         new_listings = self.cmc_client.get_new_listings(days, limit)

    #         analyzed_listings = []
    #         for listing in new_listings:
    #             try:
    #                 symbol = f"{listing['symbol']}USDT"

    #                 # Análisis completo si está en Binance
    #                 if self.client.is_valid_symbol(symbol):
    #                     recommendation = self.analyze_trading_opportunity(symbol)
    #                     market_data = self._get_market_data(symbol)

    #                     analyzed_listings.append({
    #                         **listing,
    #                         'trading_recommendation': recommendation,
    #                         'market_data': market_data,
    #                         'analysis_summary': self._generate_listing_summary(listing, recommendation, market_data),
    #                         'risk_metrics': self._calculate_risk_metrics(listing, market_data),
    #                         'trading_metrics': {
    #                             'binance_available': True,
    #                             'volume_24h': market_data.get('volume_24h', 0),
    #                             'price_change': market_data.get('price_change_24h', 0),
    #                             'volatility': market_data.get('volatility', 0)
    #                         }
    #                     })
    #             except Exception as e:
    #                 print(ConsoleColors.warning(f"Error analizando {listing['symbol']}: {str(e)}"))
    #                 continue

    #         # Ordenar por potencial y riesgo
    #         return self._sort_analyzed_listings(analyzed_listings)

    #     except Exception as e:
    #         print(ConsoleColors.error(f"Error analizando nuevos listings: {str(e)}"))
    #         return []

    def _generate_listing_summary(
        self, listing: Dict, recommendation, market_data: Dict
    ) -> Dict:
        """Genera un resumen del análisis para nuevos listings"""
        try:
            return {
                "potential_score": self._calculate_potential_score(
                    listing, market_data
                ),
                "risk_level": self._calculate_risk_level(listing, market_data),
                "recommendation": (
                    "FAVORABLE"
                    if recommendation and recommendation.signal.value == "COMPRAR"
                    else "DESFAVORABLE"
                ),
                "key_metrics": {
                    "market_cap": listing["market_cap"],
                    "volume_ratio": (
                        market_data.get("volume_24h", 0) / listing["volume_24h"]
                        if listing["volume_24h"]
                        else 0
                    ),
                    "price_stability": self._calculate_price_stability(market_data),
                },
                "warning_flags": self._get_warning_flags(listing, market_data),
            }
        except Exception as e:
            print(ConsoleColors.error(f"Error generando resumen: {str(e)}"))
            return {}

    def _calculate_risk_metrics(self, listing: Dict, market_data: Dict) -> Dict:
        """Calcula métricas de riesgo detalladas"""
        try:
            volatility = market_data.get("volatility", 0)
            volume_24h = market_data.get("volume_24h", 0)
            price_change = market_data.get("price_change_24h", 0)

            return {
                "volatility_risk": min(volatility / 0.1, 1.0),  # Normalizado a 10%
                "volume_risk": 1.0
                - min(volume_24h / 1000000, 1.0),  # Normalizado a $1M
                "price_stability_risk": min(
                    abs(price_change) / 50, 1.0
                ),  # Normalizado a 50%
                "market_cap_risk": 1.0
                - min(listing["market_cap"] / 10000000, 1.0),  # Normalizado a $10M
                "overall_risk": self._calculate_overall_risk(listing, market_data),
            }
        except Exception as e:
            print(ConsoleColors.error(f"Error calculando métricas de riesgo: {str(e)}"))
            return {}

    def _sort_analyzed_listings(self, listings: List[Dict]) -> List[Dict]:
        """Ordena los listings analizados por potencial y riesgo"""
        try:
            return sorted(
                listings,
                key=lambda x: (
                    x["analysis_summary"]["potential_score"],
                    -x["risk_metrics"]["overall_risk"],
                ),
                reverse=True,
            )
        except Exception as e:
            print(ConsoleColors.error(f"Error ordenando listings: {str(e)}"))
            return listings

    def _calculate_potential_score(self, listing: Dict, market_data: Dict) -> float:
        """Calcula score de potencial de una nueva listing"""
        try:
            # Factores positivos
            volume_score = min(market_data.get("volume_24h", 0) / 1000000, 1.0) * 0.3
            market_cap_score = min(listing["market_cap"] / 10000000, 1.0) * 0.2

            # Factor de crecimiento (basado en cambio de precio)
            price_change = abs(market_data.get("price_change_24h", 0))
            growth_score = min(price_change / 100, 1.0) * 0.3 if price_change > 0 else 0

            # Factor de estabilidad
            stability_score = self._calculate_price_stability(market_data) * 0.2

            return volume_score + market_cap_score + growth_score + stability_score

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando score de potencial: {str(e)}"))
            return 0.0

    def _calculate_price_stability(self, market_data: Dict) -> float:
        """Calcula la estabilidad del precio"""
        try:
            volatility = market_data.get("volatility", 0)
            return max(0, 1 - (volatility / 0.2))  # 20% volatilidad máxima considerada
        except Exception:
            return 0.0

    def _get_warning_flags(self, listing: Dict, market_data: Dict) -> List[str]:
        """Identifica señales de advertencia"""
        flags = []

        try:
            # Volatilidad excesiva
            if market_data.get("volatility", 0) > 0.2:
                flags.append("Alta volatilidad")

            # Volumen bajo
            if market_data.get("volume_24h", 0) < 100000:
                flags.append("Volumen bajo")

            # Cambio de precio extremo
            if abs(market_data.get("price_change_24h", 0)) > 50:
                flags.append("Movimiento de precio extremo")

            # Market cap muy bajo
            if listing["market_cap"] < 1000000:
                flags.append("Market cap muy bajo")

        except Exception as e:
            print(ConsoleColors.error(f"Error obteniendo warning flags: {str(e)}"))

        return flags

    def analyze_new_listings(
        self,
        days: int = 7,
        limit: int = 50,
        min_volume: float = 100000,
        max_mcap: float = 10000000,
    ) -> List[Dict]:
        """
        Analiza nuevas criptomonedas en el mercado con criterios específicos para gems

        Args:
            days: Número de días hacia atrás para buscar
            limit: Número máximo de resultados
            min_volume: Volumen mínimo en 24h (USD)
            max_mcap: Capitalización de mercado máxima (USD)
        """
        try:
            # Obtener nuevos listings
            new_listings = self.cmc_client.get_new_listings(
                days, limit * 2
            )  # Pedir más para filtrar

            analyzed_listings = []
            for listing in new_listings:
                try:
                    # Aplicar filtros básicos primero
                    if (
                        listing["volume_24h"] < min_volume
                        or listing["market_cap"] > max_mcap
                    ):
                        continue

                    symbol = f"{listing['symbol']}USDT"

                    # Verificar disponibilidad en Binance
                    if not self.client.is_valid_symbol(symbol):
                        continue

                    # Obtener datos adicionales
                    market_data = self._get_market_data(symbol)
                    social_data = self.cmc_client.get_social_stats(listing["symbol"])

                    # Enriquecer datos del listing
                    enriched_listing = {
                        **listing,
                        "market_data": market_data,
                        "social_metrics": social_data,
                        "availability": {
                            "binance": True,
                            "contract_verified": self._verify_contract(
                                listing["symbol"]
                            ),
                            "is_audited": self._check_audit_status(listing["symbol"]),
                        },
                        "community_metrics": self._get_community_metrics(
                            listing["symbol"]
                        ),
                        "risk_metrics": self._analyze_listing_risk(
                            listing, market_data
                        ),
                    }

                    analyzed_listings.append(enriched_listing)

                except Exception as e:
                    print(
                        ConsoleColors.warning(
                            f"Error analizando {listing['symbol']}: {str(e)}"
                        )
                    )
                    continue

            # Ordenar por potencial (volumen y engagement)
            return sorted(
                analyzed_listings,
                key=lambda x: (
                    x["volume_24h"],
                    x.get("social_metrics", {}).get("social_score", 0),
                ),
                reverse=True,
            )[:limit]

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando nuevos listings: {str(e)}"))
            return []

    def _verify_contract(self, symbol: str) -> bool:
        """Verifica si el contrato está verificado"""
        # Implementar verificación real aquí
        return True

    def _check_audit_status(self, symbol: str) -> bool:
        """Verifica si el proyecto está auditado"""
        # Implementar verificación real aquí
        return True

    def _get_community_metrics(self, symbol: str) -> Dict:
        """Obtiene métricas detalladas de la comunidad"""
        try:
            social_data = self.cmc_client.get_social_stats(symbol)
            return {
                "twitter_followers": social_data.get("twitter_followers", 0),
                "telegram_members": social_data.get("telegram_members", 0),
                "discord_members": social_data.get("discord_members", 0),
                "social_engagement": social_data.get("social_score", 0),
                "growth_rate": social_data.get("growth_rate", 0),
            }
        except Exception:
            return {}

    def _analyze_listing_risk(self, listing: Dict, market_data: Dict) -> Dict:
        """Analiza riesgos específicos de nuevos listings"""
        try:
            risk_analysis = {
                "volatility_risk": self._calculate_volatility_risk(market_data),
                "liquidity_risk": self._calculate_liquidity_risk(market_data),
                "concentration_risk": self._calculate_concentration_risk(listing),
                "overall_risk": 0.0,
            }

            # Calcular riesgo general
            risk_scores = [
                risk_analysis["volatility_risk"] * 0.4,
                risk_analysis["liquidity_risk"] * 0.3,
                risk_analysis["concentration_risk"] * 0.3,
            ]
            risk_analysis["overall_risk"] = sum(risk_scores)

            return risk_analysis

        except Exception:
            return {
                "volatility_risk": 1.0,
                "liquidity_risk": 1.0,
                "concentration_risk": 1.0,
                "overall_risk": 1.0,
            }

    def _calculate_volatility_risk(self, market_data: Dict) -> float:
        """Calcula el riesgo basado en volatilidad"""
        try:
            volatility = market_data.get("volatility_7d", 0)
            return min(volatility / 0.5, 1.0)  # Normalizar a 50% de volatilidad
        except Exception:
            return 1.0

    def _calculate_liquidity_risk(self, market_data: Dict) -> float:
        """Calcula el riesgo basado en liquidez"""
        try:
            volume = market_data.get("volume_24h", 0)
            return max(1 - (volume / 1000000), 0.0)  # Normalizar a $1M
        except Exception:
            return 1.0

    def _calculate_concentration_risk(self, listing: Dict) -> float:
        """Calcula el riesgo basado en concentración de holders"""
        try:
            top_holders = listing.get("top_holders_percentage", 100)
            return min(top_holders / 100, 1.0)
        except Exception:
            return 1.0

    def _get_market_data(self, symbol: str) -> Dict:
        """Obtiene datos detallados del mercado"""
        try:
            ticker = self.client.get_ticker_24h(symbol)
            market_metrics = self.client.calculate_market_metrics(symbol)

            return {
                "price": float(ticker["lastPrice"]) if ticker else 0,
                "volume_24h": float(ticker["volume"]) if ticker else 0,
                "price_change_24h": (
                    float(ticker["priceChangePercent"]) if ticker else 0
                ),
                "volatility_7d": market_metrics.get("volatility_7d", 0),
                "high_24h": float(ticker["highPrice"]) if ticker else 0,
                "low_24h": float(ticker["lowPrice"]) if ticker else 0,
            }
        except Exception:
            return {}

# core/market_analyzer.py
import numpy as np
from typing import Dict, List, Optional, Tuple
from utils.console_colors import ConsoleColors
from alerts.alert_manager import AlertManager
from models.data_classes import TradeRecommendation, TimingWindow
from models.enums import MarketTrend, TradingSignal, SignalStrength, EntryTiming

class MarketAnalyzer:
    def __init__(self, client, alert_manager: Optional[AlertManager] = None):
        self.client = client
        self.alert_manager = alert_manager
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
            "oversold_rsi": 30,
            "overbought_rsi": 70,
            "price_support": 0.05,    # 5% desde soporte
            "price_resistance": 0.05,  # 5% desde resistencia
            "volume_spike": 2.0,       # 2x del promedio
            "volume_significant": 1.5,  # 1.5x del promedio
            "consolidation_range": 0.02,  # 2% de rango
            "high_volatility": 0.05,      # 5% volatilidad
            "low_volatility": 0.01,       # 1% volatilidad
            "strong_momentum": 0.15,    # 15% movimiento
            "weak_momentum": 0.05,      # 5% movimiento
            "trend_strength": {
                "strong": 0.15,        # 15% movimiento
                "moderate": 0.08,      # 8% movimiento
                "weak": 0.03           # 3% movimiento
            },
            "volume_levels": {
                "high": 2.0,           # 2x promedio
                "moderate": 1.5,       # 1.5x promedio
                "low": 0.5             # 0.5x promedio
            }
        }

    def _generate_recommendation(self, trend: MarketTrend, volume_analysis: Dict,
                               momentum: Dict, rsi: float, price: float) -> Tuple[TradingSignal, SignalStrength, List[str]]:
        """Genera una recomendación de trading basada en el análisis"""
        reasons = []
        score = 0

        # Analizar tendencia
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

        # Analizar volumen
        if volume_analysis['is_significant'] and volume_analysis['is_increasing']:
            score += 1
            reasons.append(f"Alto volumen con tendencia creciente ({volume_analysis['ratio']:.1f}x promedio)")

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

    def analyze_trading_opportunity(self, symbol: str) -> Optional[TradeRecommendation]:
        """Analiza una oportunidad de trading específica"""
        try:
            # Obtener datos históricos
            candlesticks = self.client.get_klines(symbol, interval='4h', limit=120)
            ticker_24h = self.client.get_ticker_24h(symbol)

            if not candlesticks or len(candlesticks) < 50:
                print(ConsoleColors.warning(
                    f"Datos históricos insuficientes para {symbol}"
                ))
                return None

            if not ticker_24h:
                print(ConsoleColors.warning(
                    f"No se pudo obtener información actual de {symbol}"
                ))
                return None

            # Analizar diferentes aspectos
            trend = self._analyze_trend(candlesticks)
            volume_analysis = self._analyze_volume(candlesticks)
            momentum = self._analyze_momentum(candlesticks)
            rsi = self._calculate_rsi(candlesticks)

            # Obtener precio actual
            current_price = float(ticker_24h['lastPrice'])

            # Generar recomendación
            signal, strength, reasons = self._generate_recommendation(
                trend, volume_analysis, momentum, rsi, current_price
            )

            # Calcular niveles
            levels = self._calculate_trade_levels(current_price, trend, candlesticks)

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
            if daily_returns:
                std_dev = np.std(daily_returns)
            else:
                std_dev = 0

            # Combinar ATR y desviación estándar
            volatility = (atr / closes[-1] + std_dev) / 2

            return volatility

        except Exception as e:
            print(ConsoleColors.warning(f"Error calculando volatilidad: {str(e)}"))
            return 0.0

    def analyze_entry_timing(self, symbol: str) -> TimingWindow:
        """Analiza el mejor momento para entrar"""
        try:
            # Obtener datos necesarios
            candlesticks = self.client.get_klines(symbol, interval='1h', limit=100)
            ticker = self.client.get_ticker_24h(symbol)

            if not candlesticks or len(candlesticks) < 50:
                return TimingWindow(
                    EntryTiming.NOT_RECOMMENDED,
                    "N/A",
                    conditions=["Datos insuficientes para análisis"]
                )

            current_price = float(ticker['lastPrice'])

            # Realizar análisis
            rsi = self._calculate_rsi(candlesticks)
            volume_analysis = self._analyze_volume(candlesticks)
            support_resistance = self._calculate_support_resistance(candlesticks)
            volatility = self._calculate_volatility(candlesticks)
            pattern = self._identify_pattern(candlesticks)

            # Determinar timing
            timing, timeframe, target, confidence, conditions = self._determine_timing(
                current_price, rsi, volume_analysis, support_resistance,
                volatility, pattern
            )

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

    def _analyze_volume(self, candlesticks: List[Dict]) -> Dict:
        """Analiza el patrón de volumen"""
        try:
            volumes = [float(candle['volume']) for candle in candlesticks]

            if len(volumes) < 20:
                return {
                    'ratio': 1.0,
                    'is_significant': False,
                    'is_increasing': False,
                    'average': 0
                }

            avg_volume = np.mean(volumes[-20:])
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

    @staticmethod
    def _calculate_ema(values: List[float], period: int) -> List[float]:
        """Calcula EMA (Exponential Moving Average)"""
        ema = [values[0]]
        multiplier = 2 / (period + 1)

        for price in values[1:]:
            ema.append((price * multiplier) + (ema[-1] * (1 - multiplier)))

        return ema

    def _calculate_rsi(self, candlesticks: List[Dict], period: int = 14) -> float:
        """Calcula RSI (Relative Strength Index)"""
        try:
            closes = [float(candle['close']) for candle in candlesticks]
            deltas = np.diff(closes)

            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])

            if avg_loss == 0:
                return 100

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            return rsi

        except Exception:
            return 50  # Valor neutral en caso de error

    def _analyze_momentum(self, candlesticks: List[Dict]) -> Dict:
        """Analiza el momentum del precio"""
        try:
            closes = [float(candle['close']) for candle in candlesticks]

            if len(closes) < 42:
                return {
                    'short_term': 0,
                    'medium_term': 0,
                    'is_strong': False,
                    'is_positive': False
                }

            change_24h = ((closes[-1] - closes[-6]) / closes[-6]) * 100
            change_7d = ((closes[-1] - closes[-42]) / closes[-42]) * 100

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
    def _identify_pattern(self, candlesticks: List[Dict]) -> Dict:
        """Identifica patrones de precio"""
        try:
            closes = [float(candle['close']) for candle in candlesticks]
            highs = [float(candle['high']) for candle in candlesticks]
            lows = [float(candle['low']) for candle in candlesticks]
            opens = [float(candle['open']) for candle in candlesticks]

            if len(closes) < 5:
                return {'type': 'neutral', 'name': 'Datos insuficientes'}

            # Identificar patrones básicos
            # Tendencia alcista
            if all(closes[i] <= closes[i+1] for i in range(-5, -1)):
                return {'type': 'bullish', 'name': 'Tendencia Alcista Continua'}

            # Tendencia bajista
            if all(closes[i] >= closes[i+1] for i in range(-5, -1)):
                return {'type': 'bearish', 'name': 'Tendencia Bajista Continua'}

            # Doble fondo
            if (lows[-3] > lows[-2] and lows[-2] < lows[-1] and
                abs(lows[-3] - lows[-1]) < (highs[-2] - lows[-2]) * 0.1):
                return {'type': 'bullish', 'name': 'Doble Fondo'}

            # Doble techo
            if (highs[-3] < highs[-2] and highs[-2] > highs[-1] and
                abs(highs[-3] - highs[-1]) < (highs[-2] - lows[-2]) * 0.1):
                return {'type': 'bearish', 'name': 'Doble Techo'}

            # Martillo alcista
            for i in range(-3, 0):
                body = abs(opens[i] - closes[i])
                shadow_lower = min(opens[i], closes[i]) - lows[i]
                shadow_upper = highs[i] - max(opens[i], closes[i])
                if (shadow_lower > body * 2 and shadow_upper < body * 0.5 and
                    closes[i] > opens[i]):
                    return {'type': 'bullish', 'name': 'Martillo Alcista'}

            # Estrella fugaz (shooting star)
            for i in range(-3, 0):
                body = abs(opens[i] - closes[i])
                shadow_lower = min(opens[i], closes[i]) - lows[i]
                shadow_upper = highs[i] - max(opens[i], closes[i])
                if (shadow_upper > body * 2 and shadow_lower < body * 0.5 and
                    closes[i] < opens[i]):
                    return {'type': 'bearish', 'name': 'Estrella Fugaz'}

            # Patrón envolvente alcista
            for i in range(-3, 0):
                if (closes[i-1] < opens[i-1] and  # Vela anterior roja
                    closes[i] > opens[i] and      # Vela actual verde
                    opens[i] < closes[i-1] and    # Abre por debajo
                    closes[i] > opens[i-1]):      # Cierra por encima
                    return {'type': 'bullish', 'name': 'Patrón Envolvente Alcista'}

            # Patrón envolvente bajista
            for i in range(-3, 0):
                if (closes[i-1] > opens[i-1] and  # Vela anterior verde
                    closes[i] < opens[i] and      # Vela actual roja
                    opens[i] > closes[i-1] and    # Abre por encima
                    closes[i] < opens[i-1]):      # Cierra por debajo
                    return {'type': 'bearish', 'name': 'Patrón Envolvente Bajista'}

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
                            return {'type': 'bearish', 'name': 'Pin Bar Bajista'}
                        if lower_wick > 2 * body and upper_wick < 0.2 * total_length:
                            return {'type': 'bullish', 'name': 'Pin Bar Alcista'}

            # Inside Bar (Price Action)
            for i in range(-3, -1):
                if (highs[i] < highs[i-1] and lows[i] > lows[i-1]):
                    if closes[i] > opens[i]:
                        return {'type': 'bullish', 'name': 'Inside Bar Alcista'}
                    else:
                        return {'type': 'bearish', 'name': 'Inside Bar Bajista'}

            # Si no se identifica ningún patrón
            return {'type': 'neutral', 'name': 'Sin Patrón Identificado'}

        except Exception as e:
            print(ConsoleColors.error(f"Error identificando patrones: {str(e)}"))
            return {'type': 'neutral', 'name': 'Error en Identificación'}

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

    def _calculate_trade_levels(self, current_price: float, trend: MarketTrend,
                              candlesticks: List[Dict]) -> Dict:
        """Calcula niveles de entrada, stop loss y take profit"""
        try:
            # Calcular ATR para los niveles
            atr = self._calculate_atr(candlesticks)

            # Ajustar niveles según la tendencia y volatilidad
            if trend in [MarketTrend.STRONG_UPTREND, MarketTrend.UPTREND]:
                stop_loss = current_price - (atr * 2.5)
                take_profit = current_price + (atr * 3.5)
            elif trend in [MarketTrend.STRONG_DOWNTREND, MarketTrend.DOWNTREND]:
                stop_loss = current_price - (atr * 2)
                take_profit = current_price + (atr * 3)
            else:
                stop_loss = current_price - (atr * 2.2)
                take_profit = current_price + (atr * 3.2)

            # Ajustar con niveles de soporte/resistencia
            support_resistance = self._calculate_support_resistance(candlesticks)
            support_levels = support_resistance.get('support_levels', [])
            resistance_levels = support_resistance.get('resistance_levels', [])

            if support_levels:
                nearest_support = max([s for s in support_levels if s < current_price], default=stop_loss)
                stop_loss = max(stop_loss, nearest_support * 0.98)  # 2% debajo del soporte

            if resistance_levels:
                nearest_resistance = min([r for r in resistance_levels if r > current_price], default=take_profit)
                take_profit = min(take_profit, nearest_resistance * 1.02)  # 2% arriba de la resistencia

            # Calcular ratio riesgo/beneficio
            risk = current_price - stop_loss
            reward = take_profit - current_price
            risk_reward_ratio = reward / risk if risk > 0 else 0

            return {
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': risk_reward_ratio
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando niveles de trading: {str(e)}"))
            # Retornar niveles por defecto más conservadores para meme coins
            return {
                'stop_loss': current_price * 0.85,  # 15% por debajo
                'take_profit': current_price * 1.35,  # 35% por encima
                'risk_reward_ratio': 2.33  # (35% ganancia) / (15% pérdida)
            }

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

    def _adjust_timeframe(self, timeframe: str, multiplier: float) -> str:
        """Ajusta el timeframe según el multiplicador"""
        try:
            # Extraer números del timeframe
            import re
            numbers = re.findall(r'\d+', timeframe)
            if len(numbers) == 2:
                min_time = int(float(numbers[0]) * multiplier)
                max_time = int(float(numbers[1]) * multiplier)
                return f"{min_time}-{max_time} horas"
            return timeframe
        except Exception:
            return timeframe

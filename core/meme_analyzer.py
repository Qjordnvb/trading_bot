# core/meme_analyzer.py
from typing import Dict, List, Optional
import numpy as np  # Agregar esta importación
from utils.console_colors import ConsoleColors
from models.data_classes import TradeRecommendation, TimingWindow
from models.enums import TradingSignal, SignalStrength, EntryTiming

class MemeCoinAnalyzer:
    def __init__(self, client, market_analyzer):
        self.client = client
        self.market_analyzer = market_analyzer

        # Palabras clave mejoradas y categorizadas
        self.meme_keywords = {
            "primary": {
                "mascots": ["doge", "shib", "pepe", "floki", "inu", "cat", "kitty", "pug", "hamster"],
                "themes": ["meme", "chad", "wojak", "moon", "rocket", "safe"],
                "prefixes": ["baby", "mini", "mega", "super"],
                "trends": ["ai", "gpt", "meta", "web3", "defi"]
            },
            "secondary": {
                "community": ["army", "hands", "hold", "hodl", "fomo"],
                "mechanics": ["burn", "reward", "stake", "yield"],
                "sentiment": ["happy", "lucky", "rich", "million"]
            },
            "exclude": {
                "technical": ["chain", "swap", "protocol", "bridge", "oracle"],
                "defi": ["finance", "stake", "dao", "yield", "farm"],
                "infrastructure": ["node", "validator", "consensus"]
            }
        }

        # Filtros mejorados
        self.filters = {
            "market": {
                "min_volume_24h": 500000,
                "max_price": 1.0,
                "min_trades": 2000,
                "min_market_cap": 1000000
            },
            "volatility": {
                "min": 0.05,  # 5% mínimo de volatilidad
                "max": 0.5    # 50% máximo de volatilidad
            },
            "volume": {
                "min_volume_btc": 0.1,  # Mínimo volumen en BTC
                "max_wallet_concentration": 0.15  # Máximo 15% en top wallets
            },
            "social": {
                "min_holders": 1000,
                "min_social_score": 20
            }
        }

        # Nuevos pesos para scoring
        self.weights = {
            "market_metrics": 0.30,
            "technical_analysis": 0.25,
            "social_metrics": 0.20,
            "momentum": 0.15,
            "risk_factors": 0.10
        }

    def get_top_meme_coins(self, limit: int = 10) -> List[Dict]:
        try:
            # Obtener todos los pares de trading
            exchange_info = self.client.get_exchange_info()
            all_symbols = [s['symbol'] for s in exchange_info.get('symbols', [])]

            # Filtrar por monedas USDT
            usdt_pairs = [s for s in all_symbols if s.endswith('USDT')]

            # Analizar cada par
            meme_coins = []
            for symbol in usdt_pairs:
                base_asset = symbol[:-4].lower()  # Remover 'USDT'

                # Verificar si es meme coin
                if self._is_meme_coin(base_asset):
                    analysis = self._analyze_meme_coin(symbol)
                    if analysis:
                        meme_coins.append(analysis)

            # Ordenar por score y retornar los mejores
            return sorted(meme_coins, key=lambda x: x['total_score'], reverse=True)[:limit]

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando meme coins: {str(e)}"))
            return []

    def _is_meme_coin(self, currency: str) -> bool:
        # Verificar palabras excluidas primero
        for category, words in self.meme_keywords["exclude"].items():
            if any(word in currency for word in words):
                return False

        # Verificar palabras primarias
        primary_matches = 0
        for category, words in self.meme_keywords["primary"].items():
            if any(word in currency for word in words):
                primary_matches += 1

        if primary_matches >= 1:
            return True

        # Verificar palabras secundarias
        secondary_matches = 0
        for category, words in self.meme_keywords["secondary"].items():
            if any(word in currency for word in words):
                secondary_matches += 1

        return secondary_matches >= 2

    def _analyze_meme_coin(self, symbol: str) -> Optional[Dict]:
        try:
            # Obtener datos del mercado
            ticker = self.client.get_ticker_24h(symbol)
            if not ticker or not self._meets_basic_criteria(ticker):
                return None

            current_price = float(ticker['lastPrice'])

            # Análisis técnico
            market_metrics = self._analyze_market_metrics(symbol, ticker)
            if not market_metrics:
                return None

            technical_analysis = self._analyze_technical_factors(symbol)
            if not technical_analysis:
                return None

            momentum_analysis = self._analyze_momentum_factors(symbol)
            if not momentum_analysis:
                return None

            risk_analysis = self._analyze_risk_factors(symbol)
            if not risk_analysis:
                return None

            # Calcular scores individuales
            market_score = self._calculate_market_score(market_metrics)
            technical_score = self._calculate_technical_score(technical_analysis)
            momentum_score = self._calculate_momentum_score(momentum_analysis)
            risk_score = self._calculate_risk_score(risk_analysis)

            # Calcular score total ponderado
            total_score = (
                market_score * self.weights["market_metrics"] +
                technical_score * self.weights["technical_analysis"] +
                momentum_score * self.weights["momentum"] +
                risk_score * self.weights["risk_factors"]
            )

            return {
                'symbol': symbol,
                'price': current_price,
                'market_metrics': market_metrics,
                'technical_analysis': technical_analysis,
                'momentum_analysis': momentum_analysis,
                'risk_analysis': risk_analysis,
                'scores': {
                    'market': market_score,
                    'technical': technical_score,
                    'momentum': momentum_score,
                    'risk': risk_score
                },
                'total_score': float(total_score)  # Asegurar que sea float
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando {symbol}: {str(e)}"))
            return None

    def _meets_basic_criteria(self, ticker: Dict) -> bool:
        try:
            volume_24h = float(ticker.get('quoteVolume', 0))
            price = float(ticker.get('lastPrice', 0))
            trades = int(ticker.get('count', 0))

            return (
                volume_24h >= self.filters["market"]["min_volume_24h"] and
                price <= self.filters["market"]["max_price"] and
                trades >= self.filters["market"]["min_trades"]
            )
        except (ValueError, TypeError):
            return False

    def _analyze_market_metrics(self, symbol: str, ticker: Dict) -> Dict:
        try:
            current_price = float(ticker['lastPrice'])
            volume_24h = float(ticker['quoteVolume'])
            price_change = float(ticker['priceChangePercent'])

            # Calcular métricas adicionales
            volatility = self._calculate_volatility(symbol)
            liquidity = self._analyze_liquidity(symbol)
            holder_metrics = self._analyze_holder_distribution(symbol)

            return {
                'price': current_price,
                'volume_24h': volume_24h,
                'price_change_24h': price_change,
                'volatility': volatility,
                'liquidity': liquidity,
                'holder_metrics': holder_metrics
            }
        except Exception as e:
            print(ConsoleColors.error(f"Error en métricas de mercado para {symbol}: {str(e)}"))
            return {}

    def _analyze_technical_factors(self, symbol: str) -> Dict:
        """Analiza factores técnicos específicos para meme coins"""
        try:
            timeframes = ['15m', '1h', '4h', '1d']
            technical_data = {}

            for tf in timeframes:
                klines = self.client.get_klines(symbol, interval=tf, limit=100)
                if klines:
                    closes = np.array([float(k['close']) for k in klines])
                    highs = np.array([float(k['high']) for k in klines])
                    lows = np.array([float(k['low']) for k in klines])

                    technical_data[tf] = {
                        'trend': str(self.market_analyzer._analyze_trend(klines)),  # Convertir a string
                        'rsi': float(self.market_analyzer._calculate_rsi(klines)),  # Asegurar float
                        'volume_profile': self._analyze_volume_pattern(klines),
                        'support_resistance': self._find_key_levels(closes.tolist()),
                        'patterns': self._identify_patterns(klines)
                    }

            return technical_data

        except Exception as e:
            print(ConsoleColors.error(f"Error en análisis técnico para {symbol}: {str(e)}"))
            return {}

    def _analyze_momentum_factors(self, symbol: str) -> Dict:
        """Analiza factores de momentum"""
        try:
            klines = self.client.get_klines(symbol, interval='1h', limit=168)
            if not klines:
                return {}

            closes = np.array([float(k['close']) for k in klines])
            volumes = np.array([float(k['volume']) for k in klines])

            momentum_data = {
                'price_momentum': float(self._calculate_price_momentum(closes)),
                'volume_momentum': float(self._calculate_volume_momentum(volumes)),
                'buy_pressure': float(self._analyze_buy_pressure(klines)),
                'momentum_divergence': self._check_momentum_divergence(closes.tolist())
            }

            # Validar que todos los valores son numéricos
            for key, value in momentum_data.items():
                if key != 'momentum_divergence' and not isinstance(value, (int, float)):
                    momentum_data[key] = 0.0

            return momentum_data

        except Exception as e:
            print(ConsoleColors.error(f"Error en análisis de momentum para {symbol}: {str(e)}"))
            return {}

    def _analyze_risk_factors(self, symbol: str) -> Dict:
        """Analiza factores de riesgo"""
        try:
            risk_data = {
                'volatility_risk': float(self._calculate_volatility_risk(symbol)),
                'liquidity_risk': float(self._calculate_liquidity_risk(symbol)),
                'correlation_risk': float(self._calculate_correlation_risk(symbol)),
                'holder_concentration': self._analyze_holder_concentration(symbol)
            }

            # Validar que todos los valores son numéricos
            for key, value in risk_data.items():
                if key != 'holder_concentration' and not isinstance(value, (int, float)):
                    risk_data[key] = 0.0

            return risk_data

        except Exception as e:
            print(ConsoleColors.error(f"Error en análisis de riesgo para {symbol}: {str(e)}"))
            return {}

    def _calculate_market_score(self, metrics: Dict) -> float:
        try:
            if not isinstance(metrics, dict):
                return 0.0

            score = 0.0
            volume = float(metrics.get('volume_24h', 0))

            # Evaluar volumen
            if volume > self.filters["market"]["min_volume_24h"] * 5:
                score += 0.3
            elif volume > self.filters["market"]["min_volume_24h"] * 2:
                score += 0.2

            # Evaluar volatilidad
            volatility = float(metrics.get('volatility', 0))
            if self.filters["volatility"]["min"] <= volatility <= self.filters["volatility"]["max"]:
                score += 0.2

            # Evaluar liquidez
            liquidity = metrics.get('liquidity', {})
            if isinstance(liquidity, dict) and liquidity.get('is_healthy', False):
                score += 0.2

            # Evaluar distribución de holders
            holder_metrics = metrics.get('holder_metrics', {})
            if isinstance(holder_metrics, dict) and holder_metrics.get('is_well_distributed', False):
                score += 0.3

            return min(1.0, score)

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando market score: {str(e)}"))
            return 0.0

    def _calculate_technical_score(self, analysis: Dict) -> float:
        try:
            if not isinstance(analysis, dict):
                return 0.0

            score = 0.0
            timeframes_analyzed = 0

            for timeframe, data in analysis.items():
                if not isinstance(data, dict):
                    continue

                # Evaluar tendencia
                trend = str(data.get('trend', ''))
                if 'UPTREND' in trend:
                    score += 0.15

                # Evaluar RSI
                rsi = float(data.get('rsi', 50))
                if 40 <= rsi <= 60:
                    score += 0.1

                # Evaluar patrones
                patterns = data.get('patterns', [])
                if isinstance(patterns, list):
                    bullish_patterns = sum(1 for p in patterns if isinstance(p, dict) and p.get('type') == 'bullish')
                    if len(patterns) > 0:
                        score += 0.15 * (bullish_patterns / len(patterns))

                timeframes_analyzed += 1

            if timeframes_analyzed == 0:
                return 0.0

            return min(1.0, score / timeframes_analyzed)

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando technical score: {str(e)}"))
            return 0.0

    def _calculate_momentum_score(self, momentum: Dict) -> float:
        if not momentum:
            return 0.0

        score = 0.0

        # Evaluar momentum de precio
        price_momentum = momentum.get('price_momentum', 0)
        if price_momentum > 0:
            score += 0.3

        # Evaluar presión compradora
        buy_pressure = momentum.get('buy_pressure', 0)
        if buy_pressure > 0.6:
            score += 0.3

        # Evaluar divergencias
        if momentum.get('momentum_divergence', {}).get('is_bullish', False):
            score += 0.4

        return min(1.0, score)

    def _calculate_risk_score(self, risk_analysis: Dict) -> float:
        if not risk_analysis:
            return 0.0

        score = 1.0  # Comenzar con score máximo y restar por riesgos

        # Penalizar por alta volatilidad
        volatility_risk = risk_analysis.get('volatility_risk', 0)
        if volatility_risk > 0.7:
            score -= 0.3
        elif volatility_risk > 0.5:
            score -= 0.2

        # Penalizar por baja liquidez
        liquidity_risk = risk_analysis.get('liquidity_risk', 0)
        if liquidity_risk > 0.6:
            score -= 0.3

        # Penalizar por alta correlación con BTC
        correlation_risk = risk_analysis.get('correlation_risk', 0)
        if correlation_risk > 0.8:
            score -= 0.2

        # Penalizar por concentración de holders
        concentration_risk = risk_analysis.get('concentration_risk', 0)
        if concentration_risk > 0.5:
            score -= 0.2

        return max(0.0, score)

    def _calculate_volatility(self, symbol: str) -> float:
        """Calcula la volatilidad específica para meme coins"""
        try:
            candlesticks = self.client.get_klines(symbol, interval='1h', limit=168)  # 1 semana
            return self.market_analyzer._calculate_volatility(candlesticks)
        except Exception as e:
            print(ConsoleColors.error(f"Error calculando volatilidad para {symbol}: {str(e)}"))
            return 0.0

    def _calculate_price_momentum(self, prices: List[float]) -> float:
        """Calcula el momentum del precio"""
        try:
            if len(prices) < 24:  # Mínimo 24 horas de datos
                return 0.0

            # Calcular cambios porcentuales en diferentes periodos
            change_6h = (prices[-1] - prices[-6]) / prices[-6] * 100
            change_12h = (prices[-1] - prices[-12]) / prices[-12] * 100
            change_24h = (prices[-1] - prices[-24]) / prices[-24] * 100

            # Ponderar los cambios
            weighted_momentum = (
                change_6h * 0.5 +    # Peso mayor al corto plazo
                change_12h * 0.3 +   # Peso medio al medio plazo
                change_24h * 0.2     # Peso menor al largo plazo
            )

            return weighted_momentum

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando momentum: {str(e)}"))
            return 0.0

    def _calculate_volatility_risk(self, symbol: str) -> float:
        """Calcula el riesgo basado en la volatilidad"""
        try:
            volatility = self._calculate_volatility(symbol)

            # Normalizar el riesgo entre 0 y 1
            risk = min(volatility / self.filters["volatility"]["max"], 1.0)

            return risk

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando riesgo de volatilidad: {str(e)}"))
            return 0.5

    def _analyze_liquidity(self, symbol: str) -> Dict:
        """Analiza la liquidez del par"""
        try:
            ticker = self.client.get_ticker_24h(symbol)
            if not ticker:
                return {'is_healthy': False, 'score': 0}

            volume_24h = float(ticker['quoteVolume'])
            trades_24h = int(ticker['count'])

            # Calcular métricas de liquidez
            liquidity_score = min(volume_24h / self.filters["market"]["min_volume_24h"], 1.0)
            trade_frequency = min(trades_24h / self.filters["market"]["min_trades"], 1.0)

            return {
                'is_healthy': liquidity_score > 0.7 and trade_frequency > 0.7,
                'score': (liquidity_score + trade_frequency) / 2,
                'volume_24h': volume_24h,
                'trades_24h': trades_24h
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando liquidez para {symbol}: {str(e)}"))
            return {'is_healthy': False, 'score': 0}

    def _analyze_holder_distribution(self, symbol: str) -> Dict:
        """Analiza la distribución de holders (simulado ya que no tenemos acceso directo a esta info)"""
        try:
            # En un caso real, aquí obtendrías datos de la blockchain
            # Por ahora simulamos basándonos en volumen y trades
            ticker = self.client.get_ticker_24h(symbol)
            if not ticker:
                return {'is_well_distributed': False, 'score': 0}

            volume_24h = float(ticker['quoteVolume'])
            trades_24h = int(ticker['count'])

            # Estimación básica
            distribution_score = min(
                (volume_24h / self.filters["market"]["min_volume_24h"]) *
                (trades_24h / self.filters["market"]["min_trades"]),
                1.0
            )

            return {
                'is_well_distributed': distribution_score > 0.7,
                'score': distribution_score,
                'estimated_holders': trades_24h // 10  # Estimación muy básica
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando distribución de holders: {str(e)}"))
            return {'is_well_distributed': False, 'score': 0}

    def _analyze_volume_pattern(self, candlesticks: List[Dict]) -> Dict:
        """Analiza patrones de volumen"""
        try:
            volumes = [float(candle['volume']) for candle in candlesticks]
            closes = [float(candle['close']) for candle in candlesticks]

            # Calcular medias móviles de volumen
            vol_ma_short = sum(volumes[-5:]) / 5
            vol_ma_long = sum(volumes[-20:]) / 20

            # Detectar acumulación/distribución
            volume_trend = "accumulation" if vol_ma_short > vol_ma_long else "distribution"

            # Detectar explosión de volumen
            recent_avg = sum(volumes[-3:]) / 3
            historical_avg = sum(volumes[-30:-3]) / 27
            volume_explosion = recent_avg > historical_avg * 2

            return {
                'volume_trend': volume_trend,
                'volume_explosion': volume_explosion,
                'recent_volume_ratio': recent_avg / historical_avg,
                'volume_ma_trend': vol_ma_short / vol_ma_long
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error en análisis de patrón de volumen: {str(e)}"))
            return {
                'volume_trend': 'neutral',
                'volume_explosion': False,
                'recent_volume_ratio': 1.0,
                'volume_ma_trend': 1.0
            }

    def _calculate_volume_momentum(self, volumes: List[float]) -> float:
        """Calcula el momentum del volumen"""
        try:
            if len(volumes) < 24:
                return 0.0

            # Calcular cambios porcentuales en diferentes periodos
            recent_vol = sum(volumes[-6:]) / 6
            prev_vol = sum(volumes[-12:-6]) / 6

            volume_change = ((recent_vol - prev_vol) / prev_vol) * 100

            return volume_change

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando momentum de volumen: {str(e)}"))
            return 0.0

    def _calculate_liquidity_risk(self, symbol: str) -> float:
        """Calcula el riesgo de liquidez"""
        try:
            ticker = self.client.get_ticker_24h(symbol)
            if not ticker:
                return 1.0  # Máximo riesgo

            volume_24h = float(ticker['quoteVolume'])
            trades_24h = int(ticker['count'])

            # Normalizar métricas
            volume_risk = max(1 - (volume_24h / (self.filters["market"]["min_volume_24h"] * 10)), 0)
            trade_risk = max(1 - (trades_24h / (self.filters["market"]["min_trades"] * 5)), 0)

            # Combinar riesgos (70% volumen, 30% trades)
            total_risk = (volume_risk * 0.7) + (trade_risk * 0.3)

            return total_risk

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando riesgo de liquidez para {symbol}: {str(e)}"))
            return 1.0

    def _find_key_levels(self, prices: List[float]) -> Dict:
        """Encuentra niveles clave de precio"""
        try:
            if len(prices) < 20:
                return {
                    'resistance_levels': [],
                    'support_levels': [],
                    'current_resistance': max(prices) if prices else 0,
                    'current_support': min(prices) if prices else 0
                }

            # Identificar pivots
            pivots_high = []
            pivots_low = []

            for i in range(2, len(prices)-2):
                if prices[i] > max(prices[i-2:i] + prices[i+1:i+3]):
                    pivots_high.append(prices[i])
                if prices[i] < min(prices[i-2:i] + prices[i+1:i+3]):
                    pivots_low.append(prices[i])

            current_price = prices[-1]

            # Encontrar niveles relevantes
            resistance_levels = sorted([p for p in pivots_high if p > current_price])
            support_levels = sorted([p for p in pivots_low if p < current_price], reverse=True)

            return {
                'resistance_levels': resistance_levels[-3:] if resistance_levels else [current_price * 1.05],
                'support_levels': support_levels[:3] if support_levels else [current_price * 0.95],
                'current_resistance': min(resistance_levels) if resistance_levels else current_price * 1.05,
                'current_support': max(support_levels) if support_levels else current_price * 0.95
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error encontrando niveles clave: {str(e)}"))
            return {
                'resistance_levels': [],
                'support_levels': [],
                'current_resistance': 0,
                'current_support': 0
            }

    def _analyze_buy_pressure(self, candlesticks: List[Dict]) -> float:
        """Analiza la presión compradora"""
        try:
            if len(candlesticks) < 10:
                return 0.5

            buy_volume = 0
            total_volume = 0

            for candle in candlesticks[-10:]:
                close = float(candle['close'])
                open = float(candle['open'])
                volume = float(candle['volume'])

                if close > open:  # Vela verde
                    buy_volume += volume
                total_volume += volume

            return buy_volume / total_volume if total_volume > 0 else 0.5

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando presión compradora: {str(e)}"))
            return 0.5

    def _calculate_correlation_risk(self, symbol: str) -> float:
        """Calcula el riesgo de correlación con otros meme coins"""
        try:
            # Obtener otros meme coins populares para comparar
            meme_coins = ['DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT']
            correlations = []

            klines = self.client.get_klines(symbol, interval='1h', limit=168)
            symbol_returns = self._calculate_price_returns([float(k['close']) for k in klines])

            for coin in meme_coins:
                if coin != symbol:
                    try:
                        coin_klines = self.client.get_klines(coin, interval='1h', limit=168)
                        coin_returns = self._calculate_price_returns([float(k['close']) for k in coin_klines])

                        if len(coin_returns) == len(symbol_returns):
                            correlation = abs(np.corrcoef(symbol_returns, coin_returns)[0,1])
                            correlations.append(correlation)
                    except:
                        continue

            # Retornar el promedio de correlaciones o 0.5 si no hay datos
            return sum(correlations) / len(correlations) if correlations else 0.5

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando riesgo de correlación: {str(e)}"))
            return 0.5

    def _calculate_price_returns(self, prices: List[float]) -> List[float]:
        """Calcula retornos de precio"""
        return [((prices[i] - prices[i-1]) / prices[i-1]) for i in range(1, len(prices))]


    def _identify_patterns(self, candlesticks: List[Dict]) -> List[Dict]:
        """Identifica patrones de velas específicos para meme coins"""
        try:
            patterns = []
            opens = [float(candle['open']) for candle in candlesticks[-5:]]
            closes = [float(candle['close']) for candle in candlesticks[-5:]]
            highs = [float(candle['high']) for candle in candlesticks[-5:]]
            lows = [float(candle['low']) for candle in candlesticks[-5:]]

            # Explosión de volumen
            volumes = [float(candle['volume']) for candle in candlesticks[-5:]]
            avg_volume = sum(volumes[:-1]) / len(volumes[:-1])
            if volumes[-1] > avg_volume * 2:
                patterns.append({
                    'type': 'bullish' if closes[-1] > opens[-1] else 'bearish',
                    'name': 'Volume Explosion',
                    'strength': min(volumes[-1] / avg_volume, 5)
                })

            # Patrones de reversión
            for i in range(-3, 0):
                body = abs(opens[i] - closes[i])
                upper_wick = highs[i] - max(opens[i], closes[i])
                lower_wick = min(opens[i], closes[i]) - lows[i]

                # Martillo/Estrella fugaz
                if body > 0:
                    if lower_wick > body * 2 and upper_wick < body * 0.5:
                        patterns.append({
                            'type': 'bullish',
                            'name': 'Hammer',
                            'strength': min(lower_wick / body, 5)
                        })
                    elif upper_wick > body * 2 and lower_wick < body * 0.5:
                        patterns.append({
                            'type': 'bearish',
                            'name': 'Shooting Star',
                            'strength': min(upper_wick / body, 5)
                        })

            return patterns

        except Exception as e:
            print(ConsoleColors.error(f"Error identificando patrones: {str(e)}"))
            return []

    def _check_momentum_divergence(self, closes: List[float]) -> Dict:
        """Analiza divergencias en el momentum"""
        try:
            if len(closes) < 14:
                return {'is_bullish': False, 'strength': 0}

            # Calcular RSI
            rsi_values = []
            for i in range(len(closes) - 13):
                subset = closes[i:i+14]
                rsi = self.market_analyzer._calculate_rsi(subset)
                rsi_values.append(rsi)

            # Buscar divergencias
            price_trend = closes[-1] > closes[-14]
            rsi_trend = rsi_values[-1] > rsi_values[0]

            # Divergencia alcista: precio baja pero RSI sube
            bullish_divergence = not price_trend and rsi_trend
            # Divergencia bajista: precio sube pero RSI baja
            bearish_divergence = price_trend and not rsi_trend

            strength = abs(rsi_values[-1] - rsi_values[0]) / 100

            return {
                'is_bullish': bullish_divergence,
                'is_bearish': bearish_divergence,
                'strength': strength
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando divergencias: {str(e)}"))
            return {'is_bullish': False, 'strength': 0}

    def _analyze_holder_concentration(self, symbol: str) -> Dict:
        """Analiza la concentración de holders (simulado)"""
        try:
            volume_24h = float(self.client.get_ticker_24h(symbol)['quoteVolume'])
            trades_24h = float(self.client.get_ticker_24h(symbol)['count'])

            # Estimar distribución basada en volumen y trades
            avg_trade_size = volume_24h / trades_24h if trades_24h > 0 else 0
            estimated_holders = int(volume_24h / (avg_trade_size * 10))  # Estimación aproximada

            # Calcular concentración estimada
            concentration = 1.0 - (min(estimated_holders, 10000) / 10000)

            return {
                'concentration': concentration,
                'risk_level': 'high' if concentration > 0.7 else 'moderate' if concentration > 0.4 else 'low',
                'estimated_holders': estimated_holders,
                'avg_trade_size': avg_trade_size
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando concentración: {str(e)}"))
            return {
                'concentration': 1.0,
                'risk_level': 'high',
                'estimated_holders': 0,
                'avg_trade_size': 0
            }

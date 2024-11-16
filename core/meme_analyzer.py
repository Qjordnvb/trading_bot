# core/meme_analyzer.py
from typing import Dict, List, Optional
from utils.console_colors import ConsoleColors
from models.data_classes import TradeRecommendation, TimingWindow
from models.enums import TradingSignal, SignalStrength, EntryTiming

class MemeCoinAnalyzer:
    def __init__(self, client, market_analyzer):
        self.client = client
        self.market_analyzer = market_analyzer
        self.meme_keywords = {
            "primary": ["doge", "shib", "pepe", "floki", "meme", "inu", "cat", "elon", "chad", "baby"],
            "secondary": [
                "moon", "rocket", "safe", "shiba", "akita", "corgi",
                "wojak","kitty", "pup", "pug"
            ],
            "exclude": ["chain", "swap", "protocol", "finance", "stake", "dao"]
        }

        self.filters = {
            "min_volume_24h": 500000,
            "max_price": 1.0,
            "min_trades": 2000,
            "min_market_cap": 1000000,
            "max_volatility": 0.5
        }

    def get_top_meme_coins(self, limit: int = 10) -> List[Dict]:
        try:
            all_meme_coins = []
            exchange_info = self.client.get_exchange_info()
            tickers_24h = self.client.get_ticker_24h()
            ticker_dict = {ticker['symbol']: ticker for ticker in tickers_24h}

            for symbol_info in exchange_info.get('symbols', []):
                if not symbol_info['quoteAsset'] == 'USDT':
                    continue

                symbol = symbol_info['symbol']
                base_asset = symbol_info['baseAsset'].lower()

                if not self._is_meme_coin(base_asset):
                    continue

                ticker = ticker_dict.get(symbol)
                if not ticker or not self._meets_criteria(ticker):
                    continue

                analysis = self._analyze_meme_coin(symbol, ticker)
                if analysis:
                    all_meme_coins.append(analysis)

            # Add advanced score to each coin
            scored_coins = []
            for coin in all_meme_coins:
                try:
                    market_metrics = self.client.calculate_market_metrics(coin['symbol'])
                    advanced_score = self._calculate_advanced_score(
                        coin,
                        market_metrics,
                        coin['recommendation'],
                        coin['timing_analysis']
                    )
                    coin['advanced_score'] = advanced_score
                    scored_coins.append(coin)
                except Exception as e:
                    print(ConsoleColors.warning(f"Error scoring {coin['symbol']}: {str(e)}"))

            return sorted(scored_coins, key=lambda x: x['advanced_score'], reverse=True)[:limit]

        except Exception as e:
            print(ConsoleColors.error(f"Error en análisis de meme coins: {str(e)}"))
            return []

    def _calculate_advanced_score(self, coin: Dict, metrics: Dict,
                                recommendation: TradeRecommendation,
                                timing: TimingWindow) -> float:
        """Calcula un score avanzado para ranking de meme coins"""
        score = coin['score']

        if metrics:
            volatility = metrics.get('volatility_7d', 0)
            if volatility < self.filters['max_volatility']:
                score += 1

            price_change = metrics.get('change_24h', 0)
            if 5 < price_change < 20:
                score += 2
            elif price_change > 20:
                score += 1

        if timing:
            if timing.timing in [EntryTiming.IMMEDIATE, EntryTiming.WAIT_DIP]:
                score *= 1.2
            score *= (1 + timing.confidence)

        volume_24h = coin['volume_24h']
        if volume_24h > self.filters['min_volume_24h'] * 5:
            score *= 1.3
        elif volume_24h > self.filters['min_volume_24h'] * 2:
            score *= 1.1

        if recommendation.signal == TradingSignal.BUY:
            if recommendation.strength == SignalStrength.STRONG:
                score *= 1.5
            elif recommendation.strength == SignalStrength.MODERATE:
                score *= 1.3

        return score

    def _is_meme_coin(self, currency: str) -> bool:
        if any(word in currency for word in self.meme_keywords["exclude"]):
            return False

        primary_match = any(keyword in currency for keyword in self.meme_keywords["primary"])
        if primary_match:
            return True

        secondary_matches = sum(1 for keyword in self.meme_keywords["secondary"] if keyword in currency)
        return secondary_matches >= 2

    def _meets_criteria(self, ticker: Dict) -> bool:
        try:
            volume = float(ticker.get('quoteVolume', 0))
            price = float(ticker.get('lastPrice', 0))
            trades = int(ticker.get('count', 0))

            return (
                volume >= self.filters["min_volume_24h"] and
                price <= self.filters["max_price"] and
                trades >= self.filters["min_trades"]
            )
        except (ValueError, TypeError):
            return False

    def _analyze_meme_coin(self, symbol: str, ticker: Dict) -> Optional[Dict]:
        try:
            if float(ticker.get('quoteVolume', 0)) < self.filters["min_volume_24h"]:
                return None

            current_price = float(ticker['lastPrice'])
            recommendation = self.market_analyzer.analyze_trading_opportunity(symbol)
            timing_analysis = self.market_analyzer.analyze_entry_timing(symbol)

            if not recommendation:
                return None

            score = self._calculate_meme_score(ticker, recommendation)

            return {
                'symbol': symbol,
                'price': current_price,
                'volume_24h': float(ticker['quoteVolume']),
                'change_24h': float(ticker.get('priceChangePercent', 0)),
                'recommendation': recommendation,
                'timing_analysis': timing_analysis,
                'score': score
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando {symbol}: {str(e)}"))
            return None

    def _calculate_meme_score(self, ticker: Dict, recommendation: TradeRecommendation) -> float:
        score = 0

        signal_weight = {
            TradingSignal.BUY: 1,
            TradingSignal.HOLD: 0,
            TradingSignal.SELL: -1
        }
        score += signal_weight[recommendation.signal] * 2

        strength_weight = {
            SignalStrength.STRONG: 2,
            SignalStrength.MODERATE: 1,
            SignalStrength.WEAK: 0
        }
        score += strength_weight[recommendation.strength]

        try:
            volume = float(ticker['quoteVolume'])
            if volume > self.filters["min_volume_24h"] * 10:
                score += 2
            elif volume > self.filters["min_volume_24h"] * 5:
                score += 1
        except (ValueError, KeyError):
            pass

        try:
            change = float(ticker['priceChangePercent'])
            if change > 20:
                score += 2
            elif change > 10:
                score += 1
            elif change < -20:
                score -= 2
            elif change < -10:
                score -= 1
        except (ValueError, KeyError):
            pass

        return score

    def _calculate_advanced_score(self, coin: Dict, metrics: Dict,
                                recommendation: TradeRecommendation,
                                timing: TimingWindow) -> float:
        score = coin['score']  # Base score

        if metrics:
            volatility = metrics.get('volatility_7d', 0)
            if volatility < self.filters['max_volatility']:
                score += 1

            price_change = metrics.get('change_24h', 0)
            if 5 < price_change < 20:
                score += 2
            elif price_change > 20:
                score += 1

        if timing and timing.confidence > 0:
            if timing.timing in [EntryTiming.IMMEDIATE, EntryTiming.WAIT_DIP]:
                score *= 1.2
            score *= (1 + timing.confidence)

        volume_24h = coin['volume_24h']
        if volume_24h > self.filters['min_volume_24h'] * 5:
            score *= 1.3
        elif volume_24h > self.filters['min_volume_24h'] * 2:
            score *= 1.1

        if recommendation.signal == TradingSignal.BUY:
            if recommendation.strength == SignalStrength.STRONG:
                score *= 1.5
            elif recommendation.strength == SignalStrength.MODERATE:
                score *= 1.3

        return score

def test_meme_coin_analyzer():
    """Función para probar el análisis de meme coins"""
    try:
        client = BinanceClient(API_KEY, API_SECRET)
        market_analyzer = MarketAnalyzer(client)
        meme_analyzer = MemeCoinAnalyzer(client, market_analyzer)

        print(ConsoleColors.header("\n=== INICIANDO ANÁLISIS DE MEME COINS ==="))
        print(ConsoleColors.info("Buscando y analizando meme coins..."))

        top_memes = meme_analyzer.get_top_meme_coins()

        if not top_memes:
            print(ConsoleColors.warning("\nNo se encontraron meme coins que cumplan los criterios"))
            return

        print(ConsoleColors.success(f"\nSe encontraron {len(top_memes)} meme coins prometedoras"))
        print_meme_coin_analysis(top_memes)

    except Exception as e:
        print(ConsoleColors.error(f"\n❌ Error en análisis de meme coins: {str(e)}"))
        import traceback
        print(traceback.format_exc())

def print_meme_coin_analysis(meme_coins: List[Dict]) -> None:
    """Imprime el análisis de meme coins con niveles de precio"""
    print(ConsoleColors.header("\n=== TOP 10 MEME COINS ===\n"))

    for i, coin in enumerate(meme_coins, 1):
        try:
            print(ConsoleColors.header(f"\n{i}. {coin['symbol']}"))

            # Información básica
            print(ConsoleColors.info("Precio Actual: ") +
                  ConsoleColors.highlight(f"${coin['current_price']:.8f}"))
            print(ConsoleColors.info("Volumen 24h: ") +
                  ConsoleColors.success(f"${coin['volume_24h']:,.2f}"))
            print(ConsoleColors.info("Cambio 24h: ") +
                  ConsoleColors.price_change(coin['change_24h']))

            # Niveles de precio
            print(ConsoleColors.info("\nNiveles de Operación:"))
            print(ConsoleColors.highlight(f"Entrada: ${coin['entry_price']:.8f}"))
            print(ConsoleColors.error(f"Stop Loss: ${coin['stop_loss']:.8f}"))
            print(ConsoleColors.success(f"Take Profit: ${coin['take_profit']:.8f}"))
            print(ConsoleColors.info(f"Ratio Riesgo/Beneficio: {coin.get('risk_reward_ratio', 0):.2f}"))

            if coin.get('recommendation'):
                print(ConsoleColors.info("\nRecomendación de Trading:"))
                print(ConsoleColors.highlight(f"Señal: {coin['recommendation'].signal.value}"))
                print(ConsoleColors.highlight(f"Fuerza: {coin['recommendation'].strength.value}"))

                if coin['recommendation'].reasons:
                    print(ConsoleColors.info("\nRazones:"))
                    for reason in coin['recommendation'].reasons:
                        print(ConsoleColors.success(f"• {reason}"))

            if coin.get('timing_analysis'):
                timing = coin['timing_analysis']
                print(ConsoleColors.info("\nAnálisis de Timing:"))
                print(ConsoleColors.highlight(f"Recomendación: {timing.timing.value}"))
                print(ConsoleColors.highlight(f"Timeframe: {timing.timeframe}"))
                print(ConsoleColors.highlight(f"Confianza: {timing.confidence:.1%}"))

                if timing.conditions:
                    print(ConsoleColors.info("\nCondiciones:"))
                    for condition in timing.conditions:
                        print(ConsoleColors.success(f"• {condition}"))

            print(ConsoleColors.info("\nPuntuación General: ") +
                  ConsoleColors.highlight(f"{coin.get('advanced_score', coin['score']):.2f}"))
            print("-" * 50)

        except Exception as e:
            print(ConsoleColors.error(f"Error mostrando análisis para {coin.get('symbol', 'Unknown')}: {str(e)}"))
            continue

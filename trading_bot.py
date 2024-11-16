# trading_bot.py
from datetime import datetime
from typing import Optional, Dict, List
from config import config
from core.binance_client import BinanceClient
from core.market_analyzer import MarketAnalyzer
from core.meme_analyzer import MemeCoinAnalyzer
from alerts.alert_manager import AlertManager
from alerts.notifications import WhatsAppNotifier, MockNotifier
from monitor.market_monitor import MarketMonitor
from utils.console_colors import ConsoleColors

class TradingBot:
    def __init__(self):
        self.client = self._initialize_client()
        self.alert_manager = self._initialize_alert_system()
        self.market_analyzer = MarketAnalyzer(self.client, self.alert_manager)
        self.meme_analyzer = MemeCoinAnalyzer(self.client, self.market_analyzer)
        self.market_monitor = MarketMonitor(self.client, self.alert_manager)
        self.symbols_to_monitor = config.TRADING_CONFIG["default_symbols"]
        self.meme_coins: List[Dict] = []


    def _initialize_client(self) -> BinanceClient:
        return BinanceClient(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)

    def _initialize_alert_system(self) -> Optional[AlertManager]:
        try:
            if all([config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN,
                    config.TWILIO_FROM_NUMBER, config.ALERT_TO_NUMBER]):
                notifier = WhatsAppNotifier(
                    config.TWILIO_ACCOUNT_SID,
                    config.TWILIO_AUTH_TOKEN,
                    config.TWILIO_FROM_NUMBER,
                    config.ALERT_TO_NUMBER
                )
                print(ConsoleColors.success("✓ Sistema de alertas inicializado con WhatsApp"))
            else:
                print(ConsoleColors.warning("⚠️ Usando notificador de prueba"))
                notifier = MockNotifier()

            return AlertManager(notifier)
        except Exception as e:
            print(ConsoleColors.error(f"Error inicializando sistema de alertas: {str(e)}"))
            return None

    def run(self):
        print(ConsoleColors.header("\n=== ANÁLISIS DE MERCADO CRYPTO ==="))
        print(ConsoleColors.info("Fecha y hora: ") +
              ConsoleColors.highlight(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        try:
            self._analyze_main_pairs()
            self._analyze_meme_coins()
            self._start_monitoring()
        except KeyboardInterrupt:
            print(ConsoleColors.warning("\nDetención manual del bot"))
        except Exception as e:
            print(ConsoleColors.error(f"\nError en ejecución: {str(e)}"))
            import traceback
            print(ConsoleColors.error(traceback.format_exc()))
        finally:
            print(ConsoleColors.header("\n=== ANÁLISIS COMPLETADO ===\n"))

    def _analyze_main_pairs(self):
        print(ConsoleColors.header("\n=== ANÁLISIS DE PARES PRINCIPALES ==="))
        for symbol in self.symbols_to_monitor:
            self._analyze_symbol(symbol)

    def _analyze_meme_coins(self):
        print(ConsoleColors.header("\n=== ANÁLISIS DE MEME COINS ==="))
        try:
            self.meme_coins = self.meme_analyzer.get_top_meme_coins()
            if not self.meme_coins:
                print(ConsoleColors.warning("\nNo se encontraron meme coins que cumplan los criterios"))
                return

            print(ConsoleColors.success(f"\nSe encontraron {len(self.meme_coins)} meme coins prometedoras"))
            for coin in self.meme_coins:
                self._analyze_symbol(coin['symbol'])
        except Exception as e:
            print(ConsoleColors.error(f"Error en análisis de meme coins: {str(e)}"))

    def _analyze_symbol(self, symbol: str):
        print(ConsoleColors.header(f"\n=== ANÁLISIS DE {symbol} ==="))

        try:
            recommendation = self.market_analyzer.analyze_trading_opportunity(symbol)
            timing = self.market_analyzer.analyze_entry_timing(symbol)

            if recommendation:
                self._print_trading_recommendation(recommendation, symbol)
                if self.alert_manager:
                    self._setup_price_alerts(symbol, recommendation, timing)
                    if timing and timing.target_price:
                        self._setup_timing_alerts(symbol, timing)

            if timing:
                self._print_timing_analysis(timing)

            print("\n" + "="*50)

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando {symbol}: {str(e)}"))

    def _print_trading_recommendation(self, recommendation, symbol: str):
        current_data = self.client.get_ticker_24h(symbol)
        last_price = float(current_data['lastPrice']) if current_data and 'lastPrice' in current_data else None

        print(ConsoleColors.info("\nRecomendación de Trading:"))
        print(ConsoleColors.highlight(f"Señal: {recommendation.signal.value}"))
        print(ConsoleColors.highlight(f"Fuerza: {recommendation.strength.value}"))

        if recommendation.reasons:
            print(ConsoleColors.info("\nRazones:"))
            for reason in recommendation.reasons:
                print(ConsoleColors.success(f"• {reason}"))

        if last_price:
            print(ConsoleColors.info("\nNiveles de Precio:"))
            print(ConsoleColors.highlight(f"Precio actual: ${last_price:.6f}"))
            print(ConsoleColors.highlight(f"Entrada: ${recommendation.entry_price:,.6f}"))

            loss_percentage = ((recommendation.stop_loss - recommendation.entry_price) /
                             recommendation.entry_price * 100)
            profit_percentage = ((recommendation.take_profit - recommendation.entry_price) /
                               recommendation.entry_price * 100)

            print(ConsoleColors.error(f"Stop Loss: ${recommendation.stop_loss:,.8f} ({loss_percentage:.2f}%)"))
            print(ConsoleColors.success(f"Take Profit: ${recommendation.take_profit:,.8f} ({profit_percentage:.2f}%)"))

    def _print_timing_analysis(self, timing):
        print(ConsoleColors.info("\nAnálisis de Timing:"))
        print(ConsoleColors.highlight(f"Recomendación: {timing.timing.value}"))
        print(ConsoleColors.highlight(f"Timeframe: {timing.timeframe}"))
        print(ConsoleColors.highlight(f"Confianza: {timing.confidence:.1%}"))

        if timing.conditions:
            print(ConsoleColors.info("\nCondiciones:"))
            for condition in timing.conditions:
                print(ConsoleColors.success(f"• {condition}"))

    def _setup_price_alerts(self, symbol: str, recommendation, timing):
        """Configure price alerts with comprehensive trading analysis"""
        if not self.alert_manager:
            return

        try:
            # Get current market data
            ticker_24h = self.client.get_ticker_24h(symbol)
            market_metrics = self.client.calculate_market_metrics(symbol)
            candlesticks = self.client.get_klines(symbol, interval='4h', limit=14)

            # Basic price metrics
            current_price = float(ticker_24h['lastPrice'])
            volume_24h = float(ticker_24h['quoteVolume'])
            change_24h = float(ticker_24h['priceChangePercent'])

            # Calculate percentages and ratios
            stop_loss_percent = ((recommendation.stop_loss - current_price) / current_price) * 100
            take_profit_percent = ((recommendation.take_profit - current_price) / current_price) * 100
            risk_reward_ratio = abs(take_profit_percent / stop_loss_percent) if stop_loss_percent != 0 else 0

            # Technical analysis
            rsi = self.market_analyzer._calculate_rsi(candlesticks)
            trend = self.market_analyzer._analyze_trend(candlesticks)
            momentum = self.market_analyzer._analyze_momentum(candlesticks)
            volatility = market_metrics.get('volatility_7d', 0)

            additional_info = {
                # Trading Signal
                'signal': recommendation.signal.value,
                'strength': recommendation.strength.value,
                'reasons': recommendation.reasons,

                # Price Levels
                'current_price': current_price,
                'entry_price': recommendation.entry_price,
                'stop_loss': recommendation.stop_loss,
                'take_profit': recommendation.take_profit,
                'stop_loss_percent': stop_loss_percent,
                'take_profit_percent': take_profit_percent,

                # Market Analysis
                'change_24h': change_24h,
                'volume_24h': volume_24h,
                'volatility': volatility,
                'trend': trend.value if trend else 'N/A',
                'rsi': rsi,
                'momentum': momentum['medium_term'] if isinstance(momentum, dict) else 0,
                'risk_reward_ratio': risk_reward_ratio,

                # Timing Analysis
                'timing_recommendation': timing.timing.value,
                'timeframe': timing.timeframe,
                'confidence': timing.confidence * 100,
                'conditions': timing.conditions,
            }

            # Set up alerts for both take profit and stop loss
            self.alert_manager.add_price_alert(
                symbol=symbol,
                target_price=recommendation.take_profit,
                current_price=current_price,
                condition='above',
                additional_info=additional_info

            )

            self.alert_manager.add_price_alert(
                symbol=symbol,
                target_price=recommendation.stop_loss,
                current_price=current_price,
                condition='below',
                additional_info=additional_info
            )

        except Exception as e:
            print(ConsoleColors.error(f"Error configurando alertas para {symbol}: {str(e)}"))

    def _setup_timing_alerts(self, symbol: str, timing):
        """Configure timing-based alerts"""
        if not self.alert_manager or not timing.target_price:
            return

        try:
            # Get current market data
            ticker_24h = self.client.get_ticker_24h(symbol)
            market_metrics = self.client.calculate_market_metrics(symbol)

            current_price = float(ticker_24h['lastPrice'])

            additional_info = {
                'timing_recommendation': timing.timing.value,
                'timeframe': timing.timeframe,
                'confidence': timing.confidence * 100,
                'conditions': timing.conditions if timing.conditions else [],
                'volume_24h': float(ticker_24h['quoteVolume']),
                'change_24h': float(ticker_24h['priceChangePercent']),
                'volatility': market_metrics.get('volatility_7d', 0)
            }

            self.alert_manager.add_price_alert(
                symbol=symbol,
                target_price=timing.target_price,
                current_price=current_price,
                condition='above',
                additional_info=additional_info
            )

        except Exception as e:
            print(ConsoleColors.error(f"Error configurando alertas de timing para {symbol}: {str(e)}"))

    def _start_monitoring(self):
        all_symbols = self.symbols_to_monitor + [coin['symbol'] for coin in self.meme_coins]
        self.market_monitor.start_monitoring(all_symbols)

if __name__ == "__main__":
    bot = TradingBot()
    bot.run()

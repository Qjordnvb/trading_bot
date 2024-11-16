# # test/test_client.py
# from typing import Optional, List, Dict
# import os
# from datetime import datetime
# from core.binance_client import BinanceClient
# from core.market_analyzer import MarketAnalyzer
# from core.meme_analyzer import MemeCoinAnalyzer
# from alerts.alert_manager import AlertManager
# from alerts.notifications import WhatsAppNotifier, MockNotifier
# from utils.console_colors import ConsoleColors
# from config import config

# class TestClient:
#     def __init__(self):
#         self.alert_manager = self._initialize_alert_system()
#         self.client = BinanceClient(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
#         self.analyzer = MarketAnalyzer(self.client, self.alert_manager)
#         self.symbols_to_monitor = config.TRADING_CONFIG["default_symbols"]
#         self.meme_coins = []

#     def _initialize_alert_system(self) -> Optional[AlertManager]:
#         try:
#             if all([config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN,
#                     config.TWILIO_FROM_NUMBER, config.ALERT_TO_NUMBER]):
#                 notifier = WhatsAppNotifier(
#                     config.TWILIO_ACCOUNT_SID,
#                     config.TWILIO_AUTH_TOKEN,
#                     config.TWILIO_FROM_NUMBER,
#                     config.ALERT_TO_NUMBER
#                 )
#                 print(ConsoleColors.success("✓ Sistema de alertas inicializado con WhatsApp"))
#             else:
#                 print(ConsoleColors.warning("⚠️ Usando notificador de prueba"))
#                 notifier = MockNotifier()

#             return AlertManager(notifier)
#         except Exception as e:
#             print(ConsoleColors.error(f"Error inicializando sistema de alertas: {str(e)}"))
#             return None

#     def run_tests(self):
#         try:
#             print(ConsoleColors.header("\n=== ANÁLISIS DE MERCADO BINANCE ==="))
#             for symbol in self.symbols_to_monitor:
#                 self._analyze_symbol(symbol)
#             self._analyze_meme_coins()

#         except Exception as e:
#             print(ConsoleColors.error(f"\n❌ Error en las pruebas: {str(e)}"))
#             import traceback
#             print(ConsoleColors.error(traceback.format_exc()))

#     def _analyze_symbol(self, symbol: str):
#         print(ConsoleColors.header(f"\n=== ANÁLISIS DE {symbol} ==="))
#         try:
#             recommendation = self.analyzer.analyze_trading_opportunity(symbol)
#             if recommendation:
#                 self._print_trading_recommendation(recommendation, symbol)
#                 self._setup_price_alerts(symbol, recommendation)

#             timing = self.analyzer.analyze_entry_timing(symbol)
#             self._print_timing_analysis(timing)

#             if self.alert_manager and timing.target_price:
#                 self._setup_timing_alerts(symbol, timing)

#             print("\n" + "="*50)

#         except Exception as e:
#             print(ConsoleColors.error(f"Error analizando {symbol}: {str(e)}"))

#     def _analyze_meme_coins(self):
#         print(ConsoleColors.header("\n=== ANÁLISIS DE MEME COINS ==="))
#         try:
#             meme_analyzer = MemeCoinAnalyzer(self.client, self.analyzer)
#             self.meme_coins = meme_analyzer.get_top_meme_coins()

#             if not self.meme_coins:
#                 print(ConsoleColors.warning("\nNo se encontraron meme coins que cumplan los criterios"))
#                 return

#             print(ConsoleColors.success(f"\nSe encontraron {len(self.meme_coins)} meme coins prometedoras"))
#             for coin in self.meme_coins:
#                 self._analyze_symbol(coin['symbol'])

#         except Exception as e:
#             print(ConsoleColors.error(f"Error en análisis de meme coins: {str(e)}"))

#     def _print_trading_recommendation(self, recommendation, symbol: str):
#         current_data = self.client.get_ticker_24h(symbol)
#         last_price = float(current_data['lastPrice']) if current_data and 'lastPrice' in current_data else None

#         print(ConsoleColors.info("\nRecomendación de Trading:"))
#         print(ConsoleColors.highlight(f"Señal: {recommendation.signal.value}"))
#         print(ConsoleColors.highlight(f"Fuerza: {recommendation.strength.value}"))

#         if recommendation.reasons:
#             print(ConsoleColors.info("\nRazones:"))
#             for reason in recommendation.reasons:
#                 print(ConsoleColors.success(f"• {reason}"))

#         if last_price:
#             print(ConsoleColors.info("\nNiveles de Precio:"))
#             print(ConsoleColors.highlight(f"Precio actual: ${last_price:.8f}"))
#             print(ConsoleColors.highlight(f"Entrada: ${recommendation.entry_price:,.8f}"))

#             loss_percentage = ((recommendation.stop_loss - recommendation.entry_price) /
#                              recommendation.entry_price * 100)
#             profit_percentage = ((recommendation.take_profit - recommendation.entry_price) /
#                                recommendation.entry_price * 100)

#             print(ConsoleColors.error(f"Stop Loss: ${recommendation.stop_loss:,.8f} ({loss_percentage:.2f}%)"))
#             print(ConsoleColors.success(f"Take Profit: ${recommendation.take_profit:,.8f} ({profit_percentage:.2f}%)"))

#     def _print_timing_analysis(self, timing):
#         print(ConsoleColors.info("\nAnálisis de Timing:"))
#         print(ConsoleColors.highlight(f"Recomendación: {timing.timing.value}"))
#         print(ConsoleColors.highlight(f"Timeframe: {timing.timeframe}"))
#         print(ConsoleColors.highlight(f"Confianza: {timing.confidence:.1%}"))

#         if timing.conditions:
#             print(ConsoleColors.info("\nCondiciones:"))
#             for condition in timing.conditions:
#                 print(ConsoleColors.success(f"• {condition}"))

#     def _setup_price_alerts(self, symbol: str, recommendation):
#         if not self.alert_manager:
#             return

#         self.alert_manager.add_price_alert(
#             symbol,
#             recommendation.take_profit,
#             recommendation.entry_price,
#             'above'
#         )

#         self.alert_manager.add_price_alert(
#             symbol,
#             recommendation.stop_loss,
#             recommendation.entry_price,
#             'below'
#         )

#     def _setup_timing_alerts(self, symbol: str, timing):
#         if not self.alert_manager or not timing.target_price:
#             return

#         self.alert_manager.add_price_alert(
#             symbol,
#             timing.target_price,
#             timing.target_price * 0.99,
#             'above'
#         )

#     def run_alert_monitor(self):
#         if not self.alert_manager:
#             print(ConsoleColors.warning("Sistema de alertas no disponible"))
#             return

#         print(ConsoleColors.info("\nIniciando monitor de alertas..."))

#         while True:
#             try:
#                 market_data = self._fetch_market_data()
#                 self.alert_manager.check_alerts(market_data)
#                 time.sleep(60)

#             except KeyboardInterrupt:
#                 print(ConsoleColors.warning("\nDetención manual del monitor de alertas"))
#                 break
#             except Exception as e:
#                 print(ConsoleColors.error(f"Error en el monitor de alertas: {str(e)}"))
#                 time.sleep(60)

#     def _fetch_market_data(self) -> Dict[str, Dict]:
#         market_data = {}
#         symbols = self.symbols_to_monitor + [coin['symbol'] for coin in self.meme_coins]

#         for symbol in symbols:
#             ticker = self.client.get_ticker_24h(symbol)
#             if ticker:
#                 market_data[symbol] = {'price': float(ticker['lastPrice'])}

#         return market_data

#     def test_connection(self):
#         try:
#             print(ConsoleColors.header("\n=== PRUEBA DE CONEXIÓN CON BINANCE ==="))

#             print(ConsoleColors.header("\n▶ Probando obtención de exchange info..."))
#             exchange_info = self.client.get_exchange_info()
#             print(ConsoleColors.success("✓ Exchange info obtenida"))
#             if exchange_info:
#                 print(ConsoleColors.info("\nEjemplo de símbolos disponibles:"))
#                 for symbol in exchange_info.get('symbols', [])[:3]:
#                     print(f"  • {symbol['symbol']}")

#             print(ConsoleColors.header("\n▶ Probando obtención de ticker 24h para BTCUSDT..."))
#             ticker = self.client.get_ticker_24h("BTCUSDT")
#             if ticker:
#                 print(ConsoleColors.success("✓ Ticker obtenido"))
#                 print(ConsoleColors.info("\nDatos del ticker:"))
#                 self._print_dict(ticker)

#             print(ConsoleColors.header("\n▶ Probando obtención de klines para BTCUSDT..."))
#             klines = self.client.get_klines("BTCUSDT")
#             print(ConsoleColors.success(f"✓ Klines obtenidos: {len(klines)}"))
#             if klines:
#                 print(ConsoleColors.info("\nEjemplo de kline:"))
#                 self._print_dict(klines[0])

#         except Exception as e:
#             print(ConsoleColors.error(f"\n❌ Error en las pruebas: {str(e)}"))

#     def _print_dict(self, data: Dict, indent: int = 2):
#         for key, value in data.items():
#             if isinstance(value, (int, float)):
#                 print(f"{' ' * indent}• {key}: {value:,.8f}" if "price" in key.lower()
#                       else f"{' ' * indent}• {key}: {value:,}")
#             else:
#                 print(f"{' ' * indent}• {key}: {value}")

# if __name__ == "__main__":
#     tester = TestClient()
#     tester.test_connection()
#     tester.run_tests()
#     tester.run_alert_monitor()

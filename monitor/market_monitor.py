# monitor/market_monitor.py
import time
from typing import Dict, List, Optional
from utils.console_colors import ConsoleColors
from alerts.alert_manager import AlertManager
from core.market_analyzer import MarketAnalyzer

class MarketMonitor:
    def __init__(self, client, alert_manager: AlertManager):
        self.client = client
        self.alert_manager = alert_manager
        self.market_analyzer = MarketAnalyzer(client, alert_manager)
        self.monitoring = False

    def start_monitoring(self, symbols: List[str]):
        self.monitoring = True
        print(ConsoleColors.info("\nIniciando monitor de mercado..."))

        while self.monitoring:
            try:
                market_data = self._fetch_market_data(symbols)
                self.alert_manager.check_alerts(market_data)
                time.sleep(60)  # Check every minute
            except Exception as e:
                print(ConsoleColors.error(f"Error en monitoreo: {str(e)}"))
                time.sleep(60)

    def stop_monitoring(self):
        self.monitoring = False
        print(ConsoleColors.warning("\nDetención del monitor de mercado"))

    def _fetch_market_data(self, symbols: List[str]) -> Dict[str, Dict]:
        market_data = {}
        for symbol in symbols:
            try:
                ticker = self.client.get_ticker_24h(symbol)
                market_metrics = self.client.calculate_market_metrics(symbol)
                candlesticks = self.client.get_klines(symbol, interval='4h', limit=14)
                recommendation = self.market_analyzer.analyze_trading_opportunity(symbol)
                timing = self.market_analyzer.analyze_entry_timing(symbol)

                if all([ticker, market_metrics, candlesticks, recommendation, timing]):
                    current_price = float(ticker['lastPrice'])

                    # Calcular porcentajes
                    stop_loss_percent = ((recommendation.stop_loss - current_price) / current_price) * 100
                    take_profit_percent = ((recommendation.take_profit - current_price) / current_price) * 100

                    market_data[symbol] = {
                        # Trading Signal
                        'signal': recommendation.signal.value,
                        'strength': recommendation.strength.value,
                        'reasons': recommendation.reasons,

                        # Price Levels
                        'price': current_price,
                        'entry_price': recommendation.entry_price,
                        'stop_loss': recommendation.stop_loss,
                        'take_profit': recommendation.take_profit,
                        'stop_loss_percent': stop_loss_percent,
                        'take_profit_percent': take_profit_percent,

                        # Market Analysis
                        'volume': float(ticker['volume']),
                        'volume_24h': float(ticker['quoteVolume']),
                        'change_24h': float(ticker['priceChangePercent']),
                        'volatility': market_metrics.get('volatility_7d', 0),
                        'rsi': self.market_analyzer._calculate_rsi(candlesticks),
                        'trend': self.market_analyzer._analyze_trend(candlesticks),
                        'momentum': self._get_momentum_value(candlesticks),

                        # Timing Analysis
                        'timing_recommendation': timing.timing.value,
                        'timeframe': timing.timeframe,
                        'confidence': timing.confidence * 100,
                        'conditions': timing.conditions
                    }
            except Exception as e:
                print(ConsoleColors.error(f"Error obteniendo datos para {symbol}: {str(e)}"))
        return market_data

    def _get_momentum_value(self, candlesticks) -> float:
        try:
            momentum = self.market_analyzer._analyze_momentum(candlesticks)
            if isinstance(momentum, dict):
                return momentum.get('medium_term', 0)
            return 0
        except Exception:
            return 0

    def _process_alerts(self, symbol: str, recommendation, timing, market_data: Dict):
        try:
            current_price = market_data['price']

            # Configurar alertas solo para los niveles de take profit y stop loss
            alert_info = dict(market_data)  # Usar toda la información del market_data

            # Configurar alertas de take profit y stop loss
            if recommendation.take_profit > current_price:
                self.alert_manager.add_price_alert(
                    symbol=symbol,
                    target_price=recommendation.take_profit,
                    current_price=current_price,
                    condition='above',
                    additional_info=alert_info
                )

            if recommendation.stop_loss < current_price:
                self.alert_manager.add_price_alert(
                    symbol=symbol,
                    target_price=recommendation.stop_loss,
                    current_price=current_price,
                    condition='below',
                    additional_info=alert_info
                )

            print(ConsoleColors.success(f"Análisis y alertas actualizados para {symbol}"))

        except Exception as e:
            print(ConsoleColors.error(f"Error procesando alertas para {symbol}: {str(e)}"))

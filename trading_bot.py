from datetime import datetime, timedelta
from typing import Optional, Dict, List
from config import config
from core.binance_client import BinanceClient
from core.coinmarketcap_client import CoinMarketCapClient
from core.market_analyzer import MarketAnalyzer
from core.meme_analyzer import MemeCoinAnalyzer
from backtesting.backtest_testing import BacktestSystem
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
        self.cmc_client = CoinMarketCapClient(config.CMC_API_KEY)
        self.market_opportunities: List[Dict] = []
        self.meme_coins: List[Dict] = []
        self.new_listings: List[Dict] = []

        # self.backtest_system = BacktestSystem(
        #     market_analyzer=self.market_analyzer,
        #     meme_analyzer=self.meme_analyzer,
        #     client=self.client,
        #     config=config
        # )

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
            # Obtener y analizar oportunidades de mercado
            self._analyze_market_opportunities()

            # Analizar meme coins
            self._analyze_meme_coins()

            # Analizar nuevas listings
            self._analyze_new_listings()

            # Iniciar monitoreo solo si hay símbolos para monitorear
            self._start_monitoring()

        except KeyboardInterrupt:
            print(ConsoleColors.warning("\nDetención manual del bot"))
        except Exception as e:
            print(ConsoleColors.error(f"\nError en ejecución: {str(e)}"))
            import traceback
            print(ConsoleColors.error(traceback.format_exc()))
        finally:
            print(ConsoleColors.header("\n=== ANÁLISIS COMPLETADO ===\n"))

    # def run(self):
    #     print(ConsoleColors.header("\n=== ANÁLISIS DE MERCADO CRYPTO ==="))
    #     print(ConsoleColors.info("Fecha y hora: ") +
    #           ConsoleColors.highlight(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    #     try:
    #         # Ejecutar backtesting primero con fechas más cortas
    #         current_date = datetime.now()
    #         end_date = current_date.strftime("%Y-%m-%d")
    #         start_date = (current_date - timedelta(days=30)).strftime("%Y-%m-%d")  # últimos 30 días

    #         backtest_results = self.run_backtest(
    #             symbol="BTCUSDT",
    #             start_time=start_date,
    #             end_time=end_date,
    #             initial_capital=10000.0
    #         )

    #         # Si el backtest es exitoso, ejecutar el bot
    #         if backtest_results and backtest_results['metrics'].get('overall_score', 0) >= 0.7:
    #             # Obtener y analizar oportunidades de mercado
    #             self._analyze_market_opportunities()

    #             # Analizar meme coins
    #             self._analyze_meme_coins()

    #             # Analizar nuevas listings
    #             self._analyze_new_listings()

    #             # Iniciar monitoreo
    #             self._start_monitoring()
    #         else:
    #             print(ConsoleColors.warning("\nOptimiza la estrategia antes de ejecutar el bot en vivo"))

    #     except KeyboardInterrupt:
    #         print(ConsoleColors.warning("\nDetención manual del bot"))
    #     except Exception as e:
    #         print(ConsoleColors.error(f"\nError en ejecución: {str(e)}"))
    #         import traceback
    #         print(ConsoleColors.error(traceback.format_exc()))
    #     finally:
    #         self.cleanup()
    #         print(ConsoleColors.header("\n=== ANÁLISIS COMPLETADO ===\n"))

    def _analyze_market_opportunities(self):
        print(ConsoleColors.header("\n=== ANÁLISIS DE OPORTUNIDADES DE MERCADO ==="))
        try:
            # Obtener oportunidades de mercado de CMC
            self.market_opportunities = self.cmc_client.get_market_opportunities(
                limit=config.TRADING_CONFIG["market_data"]["top_coins"],
                include_trending=config.TRADING_CONFIG["market_data"]["include_trending"]
            )

            if not self.market_opportunities:
                print(ConsoleColors.warning("\nNo se encontraron oportunidades de mercado"))
                return

            print(ConsoleColors.success(f"\nSe encontraron {len(self.market_opportunities)} oportunidades:"))

            # Analizar cada oportunidad
            for opportunity in self.market_opportunities:
                self._analyze_market_opportunity(opportunity)

        except Exception as e:
            print(ConsoleColors.error(f"Error en análisis de oportunidades: {str(e)}"))

    def _analyze_market_opportunity(self, opportunity: Dict):
        """Analiza una oportunidad de mercado específica"""
        # Obtener el símbolo base y añadir USDT para Binance
        base_symbol = opportunity['symbol']
        binance_symbol = f"{base_symbol}USDT"

        print(ConsoleColors.header(f"\n=== ANÁLISIS DE {base_symbol} ==="))
        print(ConsoleColors.info("Datos de Mercado:"))
        print(ConsoleColors.highlight(f"Rank: #{opportunity['rank']}"))
        print(ConsoleColors.highlight(f"Cambio 24h: {opportunity['percent_change_24h']:.2f}%"))
        print(ConsoleColors.highlight(f"Volumen: ${opportunity['volume_24h']:,.2f}"))

        if opportunity.get('type') == 'trending':
            print(ConsoleColors.success("¡Moneda en tendencia!"))

        self._analyze_symbol(binance_symbol)

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

    def _analyze_new_listings(self):
        """Analiza nuevas listings buscando potenciales gems basado en criterios específicos"""
        print(ConsoleColors.header("\n=== ANÁLISIS DE NUEVAS CRYPTOJOYAS ==="))
        try:
            # Obtener nuevos listings con criterios más específicos
            potential_gems = self.market_analyzer.analyze_new_listings(
                days=7,          # Últimos 7 días
                limit=50,        # Aumentamos el límite inicial para tener más candidatos
                min_volume=100000,  # Volumen mínimo de $100k
                max_mcap=10000000   # Market cap máximo de $10M para potencial x100
            )

            if not potential_gems:
                print(ConsoleColors.warning("\nNo se encontraron nuevas listings con potencial x100"))
                return

            # Filtrar y analizar basado en criterios de gems
            analyzed_gems = []
            for gem in potential_gems:
                gem_analysis = self._analyze_gem_potential(gem)
                if gem_analysis['total_score'] >= 7.0:  # Score mínimo de 7/10
                    analyzed_gems.append({**gem, **gem_analysis})

            if analyzed_gems:
                print(ConsoleColors.success(f"\nSe encontraron {len(analyzed_gems)} cryptojoyas potenciales:"))
                self._print_gem_analysis(analyzed_gems)
            else:
                print(ConsoleColors.warning("\nNinguna listing cumple los criterios de cryptojoya"))

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando nuevas gems: {str(e)}"))


    def _analyze_gem_potential(self, listing: Dict) -> Dict:
        """Analiza el potencial de una cryptojoya basado en múltiples criterios"""
        try:
            symbol = f"{listing['symbol']}USDT"

            # 1. Análisis Fundamental (30%)
            fundamental_score = self._analyze_fundamentals(listing)

            # 2. Análisis de Comunidad (25%)
            community_score = self._analyze_community(listing)

            # 3. Análisis de Mercado (25%)
            market_score = self._analyze_market_metrics(listing)

            # 4. Análisis de Riesgo (20%)
            risk_score = self._analyze_gem_risk(listing)

            # Calcular score total (0-10)
            total_score = (
                fundamental_score * 0.30 +
                community_score * 0.25 +
                market_score * 0.25 +
                risk_score * 0.20
            ) * 10

            flags = []
            if total_score >= 8.0:
                potential = "ALTO (Posible x100)"
            elif total_score >= 7.0:
                potential = "MEDIO (Posible x20-x50)"
            else:
                potential = "BAJO (Posible x2-x10)"

            # Generar warning flags
            if listing.get('volume_24h', 0) < 500000:
                flags.append("⚠️ Volumen bajo")
            if listing.get('holder_count', 0) < 1000:
                flags.append("⚠️ Pocos holders")
            if listing.get('liquidity', 0) < 100000:
                flags.append("⚠️ Baja liquidez")

            return {
                'total_score': total_score,
                'fundamental_score': fundamental_score,
                'community_score': community_score,
                'market_score': market_score,
                'risk_score': risk_score,
                'potential': potential,
                'warning_flags': flags,
                'analysis_summary': self._generate_gem_summary(listing, total_score)
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando potencial de {listing['symbol']}: {str(e)}"))
            return {
                'total_score': 0,
                'potential': "ERROR",
                'warning_flags': ["Error en análisis"]
            }

    def _analyze_fundamentals(self, listing: Dict) -> float:
        """Analiza fundamentales del proyecto"""
        score = 0.0
        try:
            # Verificar utilidad y caso de uso
            if listing.get('category') in ['defi', 'gaming', 'web3', 'ai']:
                score += 0.3

            # Verificar tokenomics
            if listing.get('max_supply'):
                score += 0.2
            if listing.get('burn_mechanism'):
                score += 0.2

            # Verificar equipo y transparencia
            if listing.get('is_audited'):
                score += 0.2
            if listing.get('team_verified'):
                score += 0.1

        except Exception as e:
            print(ConsoleColors.error(f"Error en análisis fundamental: {str(e)}"))

        return min(score, 1.0)

    def _analyze_community(self, listing: Dict) -> float:
        """Analiza métricas de comunidad"""
        score = 0.0
        try:
            # Analizar redes sociales
            social_data = self.cmc_client.get_social_stats(listing['symbol'])

            # Twitter Score (40%)
            followers = social_data.get('twitter_followers', 0)
            if followers > 100000: score += 0.4
            elif followers > 50000: score += 0.3
            elif followers > 10000: score += 0.2

            # Telegram/Discord Score (30%)
            members = social_data.get('chat_members', 0)
            if members > 50000: score += 0.3
            elif members > 20000: score += 0.2
            elif members > 5000: score += 0.1

            # Engagement Score (30%)
            engagement = social_data.get('social_engagement', 0)
            if engagement > 0.8: score += 0.3
            elif engagement > 0.5: score += 0.2
            elif engagement > 0.3: score += 0.1

        except Exception as e:
            print(ConsoleColors.error(f"Error en análisis de comunidad: {str(e)}"))

        return min(score, 1.0)

    def _analyze_market_metrics(self, listing: Dict) -> float:
        """Analiza métricas de mercado"""
        score = 0.0
        try:
            # Market Cap Score (30%)
            mcap = listing.get('market_cap', 0)
            if mcap < 1000000: score += 0.3  # Menos de $1M
            elif mcap < 5000000: score += 0.2  # Menos de $5M
            elif mcap < 10000000: score += 0.1  # Menos de $10M

            # Volumen Score (40%)
            volume = listing.get('volume_24h', 0)
            if volume > 1000000: score += 0.4
            elif volume > 500000: score += 0.3
            elif volume > 100000: score += 0.2

            # Liquidez Score (30%)
            liquidity = listing.get('liquidity', 0)
            if liquidity > 500000: score += 0.3
            elif liquidity > 200000: score += 0.2
            elif liquidity > 50000: score += 0.1

        except Exception as e:
            print(ConsoleColors.error(f"Error en análisis de mercado: {str(e)}"))

        return min(score, 1.0)

    def _analyze_gem_risk(self, listing: Dict) -> float:
        """Analiza el riesgo específico para gems"""
        risk_score = 1.0  # Comenzar con máximo score (menor riesgo)
        try:
            # Concentración de holders (40% del riesgo)
            top_holders = listing.get('top_holders_percentage', 0)
            if top_holders > 50: risk_score -= 0.4
            elif top_holders > 30: risk_score -= 0.2

            # Liquidez bloqueada (30% del riesgo)
            locked_liquidity = listing.get('locked_liquidity_percentage', 0)
            if locked_liquidity < 50: risk_score -= 0.3
            elif locked_liquidity < 80: risk_score -= 0.15

            # Auditoría y seguridad (30% del riesgo)
            if not listing.get('is_audited'):
                risk_score -= 0.15
            if not listing.get('contract_verified'):
                risk_score -= 0.15

        except Exception as e:
            print(ConsoleColors.error(f"Error en análisis de riesgo: {str(e)}"))
            risk_score = 0.0

        return max(risk_score, 0.0)

    def _generate_gem_summary(self, listing: Dict, total_score: float) -> Dict:
        """Genera un resumen detallado para una potencial cryptojoya"""
        return {
            'name': listing['name'],
            'symbol': listing['symbol'],
            'score': total_score,
            'key_metrics': {
                'market_cap': listing.get('market_cap', 0),
                'volume_24h': listing.get('volume_24h', 0),
                'holders': listing.get('holder_count', 0),
                'liquidity': listing.get('liquidity', 0)
            },
            'community': {
                'twitter_followers': listing.get('twitter_followers', 0),
                'telegram_members': listing.get('telegram_members', 0),
                'social_engagement': listing.get('social_engagement', 0)
            },
            'risk_factors': {
                'top_holders': listing.get('top_holders_percentage', 0),
                'locked_liquidity': listing.get('locked_liquidity_percentage', 0),
                'is_audited': listing.get('is_audited', False),
                'contract_verified': listing.get('contract_verified', False)
            }
        }

    def _print_gem_analysis(self, gems: List[Dict]):
        """Imprime el análisis detallado de las cryptojoyas encontradas"""
        for gem in gems:
            print(ConsoleColors.header(f"\n=== {gem['name']} ({gem['symbol']}) ==="))
            print(ConsoleColors.highlight(f"Score Total: {gem['total_score']:.1f}/10"))
            print(ConsoleColors.highlight(f"Potencial: {gem['potential']}"))

            print(ConsoleColors.info("\nScores por Categoría:"))
            print(ConsoleColors.success(f"• Fundamentales: {gem['fundamental_score']*10:.1f}/10"))
            print(ConsoleColors.success(f"• Comunidad: {gem['community_score']*10:.1f}/10"))
            print(ConsoleColors.success(f"• Mercado: {gem['market_score']*10:.1f}/10"))
            print(ConsoleColors.success(f"• Riesgo: {gem['risk_score']*10:.1f}/10"))

            if gem['warning_flags']:
                print(ConsoleColors.warning("\nAdvertencias:"))
                for flag in gem['warning_flags']:
                    print(ConsoleColors.error(f"• {flag}"))

            summary = gem['analysis_summary']
            print(ConsoleColors.info("\nMétricas Clave:"))
            print(ConsoleColors.highlight(f"• Market Cap: ${summary['key_metrics']['market_cap']:,.2f}"))
            print(ConsoleColors.highlight(f"• Volumen 24h: ${summary['key_metrics']['volume_24h']:,.2f}"))
            print(ConsoleColors.highlight(f"• Holders: {summary['key_metrics']['holders']:,}"))
            print(ConsoleColors.highlight(f"• Liquidez: ${summary['key_metrics']['liquidity']:,.2f}"))

            print("\n" + "="*50)

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

    def _print_trading_recommendation(self, recommendation, symbol: str, market_data: Dict = None):
        current_data = self.client.get_ticker_24h(symbol)
        last_price = float(current_data['lastPrice']) if current_data and 'lastPrice' in current_data else None

        print(ConsoleColors.info("\nRecomendación de Trading:"))
        print(ConsoleColors.highlight(f"Señal: {recommendation.signal.value}"))
        print(ConsoleColors.highlight(f"Fuerza: {recommendation.strength.value}"))

        if market_data:
            print(ConsoleColors.info("\nMétricas de Mercado:"))
            print(ConsoleColors.highlight(f"Market Cap: ${market_data.get('market_cap', 0):,.2f}"))
            print(ConsoleColors.highlight(f"Dominancia: {market_data.get('market_dominance', 0):.2f}%"))

        if recommendation.reasons:
            print(ConsoleColors.info("\nRazones:"))
            for reason in recommendation.reasons:
                print(ConsoleColors.success(f"• {reason}"))

        if last_price:
            print(ConsoleColors.info("\nNiveles de Precio:"))
            print(ConsoleColors.highlight(f"Precio actual: ${last_price:.8f}"))
            print(ConsoleColors.highlight(f"Entrada: ${recommendation.entry_price:,.8f}"))

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

    def _start_monitoring(self):
        """Inicia el monitoreo de los símbolos analizados"""
        try:
            # Obtener símbolos de todas las fuentes
            market_symbols = [opp['symbol'] + 'USDT' for opp in self.market_opportunities]
            meme_symbols = [coin['symbol'] for coin in self.meme_coins]
            new_listing_symbols = [f"{listing['symbol']}USDT" for listing in self.new_listings
                                 if listing.get('trading_recommendation')]

            # Combinar todos los símbolos únicos
            all_symbols = list(set(market_symbols + meme_symbols + new_listing_symbols))

            # Filtrar símbolos válidos en Binance
            valid_symbols = [symbol for symbol in all_symbols if self.client.is_valid_symbol(symbol)]

            if valid_symbols:
                print(ConsoleColors.info(f"\nIniciando monitoreo de {len(valid_symbols)} símbolos:"))
                for symbol in valid_symbols:
                    print(ConsoleColors.highlight(f"• {symbol}"))
                self.market_monitor.start_monitoring(valid_symbols)
            else:
                print(ConsoleColors.warning("\nNo hay símbolos válidos para monitorear"))

        except Exception as e:
            print(ConsoleColors.error(f"Error iniciando monitoreo: {str(e)}"))

    def _setup_price_alerts(self, symbol: str, recommendation, timing, risk_metrics: Dict = None):
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
                'volatility': market_metrics.get('volatility_7d', 0),

                # Timing Analysis
                'timing_recommendation': timing.timing.value if timing else None,
                'timeframe': timing.timeframe if timing else None,
                'confidence': timing.confidence * 100 if timing else 0,
                'conditions': timing.conditions if timing else [],

                # Risk Metrics (para nuevos listings)
                'risk_metrics': risk_metrics if risk_metrics else {}
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

    def get_market_opportunities(self) -> List[Dict]:
        """Obtiene oportunidades de mercado de CMC y las valida con Binance"""
        try:
            opportunities = []
            cmc_opportunities = self.cmc_client.get_market_opportunities(
                limit=config.TRADING_CONFIG["market_data"]["top_coins"],
                include_trending=config.TRADING_CONFIG["market_data"]["include_trending"]
            )

            # Filtrar solo las monedas que existen en Binance
            for opp in cmc_opportunities:
                binance_symbol = f"{opp['symbol']}USDT"
                try:
                    # Verificar si el par existe en Binance
                    test_data = self.client.get_ticker_24h(binance_symbol)
                    if test_data:
                        opportunities.append(opp)
                except Exception:
                    continue

            return opportunities
        except Exception as e:
            print(ConsoleColors.error(f"Error obteniendo oportunidades de mercado: {str(e)}"))
            return []

    def get_monitored_symbols(self) -> List[str]:
        """Obtiene la lista de todos los símbolos monitoreados"""
        symbols = set()

        # Añadir símbolos de oportunidades de mercado
        symbols.update(opp['symbol'] + 'USDT' for opp in self.market_opportunities)

        # Añadir símbolos de meme coins
        symbols.update(coin['symbol'] for coin in self.meme_coins)

        # Añadir símbolos de nuevos listings
        symbols.update(f"{listing['symbol']}USDT" for listing in self.new_listings
                      if listing.get('trading_recommendation'))

        # Filtrar solo símbolos válidos en Binance
        return [symbol for symbol in symbols if self.client.is_valid_symbol(symbol)]

    def get_active_alerts(self) -> List[Dict]:
        """Obtiene la lista de alertas activas"""
        if self.alert_manager:
            return [
                {
                    'symbol': alert.symbol,
                    'type': alert.type,
                    'target_price': alert.target_price,
                    'current_price': alert.current_price,
                    'condition': alert.condition,
                    'is_triggered': alert.is_triggered
                }
                for alert in self.alert_manager.alerts
                if not alert.is_triggered
            ]
        return []

    def update_all_alerts(self):
        """Actualiza todas las alertas activas"""
        try:
            for symbol in self.get_monitored_symbols():
                current_price = float(self.client.get_ticker_price(symbol)['price'])
                self.alert_manager.check_price_alerts(symbol, current_price)
        except Exception as e:
            print(ConsoleColors.error(f"Error actualizando alertas: {str(e)}"))

    def get_analysis_summary(self) -> Dict:
        """Obtiene un resumen del análisis actual"""
        return {
            'market_opportunities': len(self.market_opportunities),
            'meme_coins': len(self.meme_coins),
            'new_listings': len(self.new_listings),
            'monitored_symbols': len(self.get_monitored_symbols()),
            'active_alerts': len(self.get_active_alerts()),
            'last_update': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def get_risk_metrics(self, symbol: str) -> Dict:
        """Obtiene métricas de riesgo para un símbolo específico"""
        try:
            market_data = self.client.calculate_market_metrics(symbol)
            cmc_data = self.cmc_client.get_market_metrics(symbol.replace('USDT', ''))

            return {
                'volatility': market_data.get('volatility_7d', 0),
                'volume_24h': market_data.get('volume_24h', 0),
                'market_cap': cmc_data.get('market_cap', 0),
                'market_dominance': cmc_data.get('market_dominance', 0),
                'risk_score': self._calculate_risk_score(market_data, cmc_data)
            }
        except Exception as e:
            print(ConsoleColors.error(f"Error obteniendo métricas de riesgo para {symbol}: {str(e)}"))
            return {}

    def _calculate_risk_score(self, market_data: Dict, cmc_data: Dict) -> float:
        """Calcula un score de riesgo basado en múltiples métricas"""
        try:
            risk_score = 0.0
            weight = 0.0

            # Factor de volatilidad (30%)
            if 'volatility_7d' in market_data:
                volatility = market_data['volatility_7d']
                risk_score += (min(volatility, 0.2) / 0.2) * 0.3
                weight += 0.3

            # Factor de volumen (30%)
            if 'volume_24h' in market_data:
                volume = market_data['volume_24h']
                volume_score = min(volume / 1000000, 1.0)  # Normalizar a $1M
                risk_score += (1 - volume_score) * 0.3
                weight += 0.3

            # Factor de market cap (20%)
            if 'market_cap' in cmc_data:
                mcap = cmc_data['market_cap']
                mcap_score = min(mcap / 100000000, 1.0)  # Normalizar a $100M
                risk_score += (1 - mcap_score) * 0.2
                weight += 0.2

            # Factor de dominancia (20%)
            if 'market_dominance' in cmc_data:
                dominance = cmc_data['market_dominance']
                risk_score += (1 - min(dominance / 1.0, 1.0)) * 0.2
                weight += 0.2

            return risk_score / weight if weight > 0 else 1.0

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando risk score: {str(e)}"))
            return 1.0

    def cleanup(self):
        """Limpia recursos y guarda estado antes de cerrar"""
        try:
            if self.alert_manager:
                self.alert_manager.save_alerts()
            print(ConsoleColors.success("\nEstado guardado correctamente"))
        except Exception as e:
            print(ConsoleColors.error(f"Error en limpieza: {str(e)}"))

    def run_backtest(self, symbol: str, start_time: str, end_time: str, initial_capital: float = 10000.0):
        """
        Ejecuta el backtesting para un símbolo específico
        """
        print(ConsoleColors.header(f"\n=== INICIANDO BACKTEST PARA {symbol} ==="))
        try:
            results = self.backtest_system.run_backtest(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                initial_capital=initial_capital
            )

            # Analizar resultados
            if results['metrics'].get('overall_score', 0) >= 0.7:
                print(ConsoleColors.success("\n✓ Estrategia validada con éxito"))
                self._setup_validated_strategy(symbol, results)
            else:
                print(ConsoleColors.warning("\n⚠️ La estrategia necesita optimización"))
                self._suggest_strategy_improvements(results)

            return results

        except Exception as e:
            print(ConsoleColors.error(f"Error en backtesting: {str(e)}"))
            return None

    def _setup_validated_strategy(self, symbol: str, backtest_results: Dict):
        """
        Configura la estrategia validada para trading en vivo
        """
        try:
            # Extraer parámetros optimizados
            optimal_params = backtest_results.get('optimal_parameters', {})

            # Actualizar configuración del analizador
            if optimal_params:
                self.market_analyzer.update_strategy_parameters(optimal_params)

            # Configurar alertas basadas en resultados del backtest
            if self.alert_manager:
                self._setup_backtest_based_alerts(symbol, backtest_results)

            print(ConsoleColors.success("\nEstrategia configurada para trading en vivo"))
            self._print_strategy_summary(backtest_results)

        except Exception as e:
            print(ConsoleColors.error(f"Error configurando estrategia: {str(e)}"))

    def _suggest_strategy_improvements(self, results: Dict):
        """
        Sugiere mejoras basadas en resultados del backtest
        """
        try:
            metrics = results['metrics']

            print(ConsoleColors.info("\nSugerencias de Mejora:"))

            if metrics.get('win_rate', 0) < 0.5:
                print(ConsoleColors.warning("• Mejorar precisión de señales"))
                print("  - Ajustar filtros de validación")
                print("  - Aumentar confirmaciones requeridas")

            if metrics.get('profit_factor', 0) < 2:
                print(ConsoleColors.warning("• Optimizar gestión de riesgo"))
                print("  - Ajustar ratios de riesgo/beneficio")
                print("  - Revisar niveles de take profit")

            if metrics.get('max_drawdown', 0) > 20:
                print(ConsoleColors.warning("• Reducir drawdown"))
                print("  - Implementar stops más ajustados")
                print("  - Reducir tamaño de posiciones")

        except Exception as e:
            print(ConsoleColors.error(f"Error generando sugerencias: {str(e)}"))

    def _print_strategy_summary(self, results: Dict):
        """
        Imprime resumen de la estrategia validada
        """
        try:
            metrics = results['metrics']
            print(ConsoleColors.header("\n=== RESUMEN DE ESTRATEGIA VALIDADA ==="))

            print(ConsoleColors.info("\nMétricas Principales:"))
            print(f"Win Rate: {metrics.get('win_rate', 0):.2f}%")
            print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            print(f"Sharpe Ratio: {metrics.get('risk_metrics', {}).get('sharpe_ratio', 0):.2f}")

            print(ConsoleColors.info("\nParámetros Optimizados:"))
            for param, value in results.get('optimal_parameters', {}).items():
                print(f"{param}: {value}")

        except Exception as e:
            print(ConsoleColors.error(f"Error imprimiendo resumen: {str(e)}"))

    def _setup_backtest_based_alerts(self, symbol: str, backtest_results: Dict):
        """
        Configura alertas basadas en resultados del backtest
        """
        try:
            metrics = backtest_results['metrics']

            # Configurar alertas de precio
            if 'optimal_levels' in backtest_results:
                levels = backtest_results['optimal_levels']
                current_price = float(self.client.get_ticker_price(symbol)['price'])

                for level in levels:
                    self.alert_manager.add_price_alert(
                        symbol=symbol,
                        target_price=level['price'],
                        current_price=current_price,
                        condition=level['condition'],
                        additional_info={
                            'backtest_confidence': metrics.get('overall_score', 0),
                            'optimal_parameters': backtest_results.get('optimal_parameters', {}),
                            'level_type': level['type']
                        }
                    )

        except Exception as e:
            print(ConsoleColors.error(f"Error configurando alertas: {str(e)}"))

if __name__ == "__main__":
    bot = TradingBot()
    try:
        bot.run()
    except KeyboardInterrupt:
        print(ConsoleColors.warning("\nDetención manual del bot"))
    except Exception as e:
        print(ConsoleColors.error(f"\nError en ejecución: {str(e)}"))
    finally:
        bot.cleanup()
# if __name__ == "__main__":
#     bot = TradingBot()
#     try:
#         # Ejecutar backtesting primero
#         backtest_results = bot.run_backtest(
#             symbol="BTCUSDT",
#             start_time="2023-01-01",
#             end_time="2023-12-31",
#             initial_capital=10000.0
#         )

#         # Si el backtest es exitoso, ejecutar el bot
#         if backtest_results and backtest_results['metrics'].get('overall_score', 0) >= 0.7:
#             bot.run()
#         else:
#             print(ConsoleColors.warning("\nOptimiza la estrategia antes de ejecutar el bot en vivo"))

#     except KeyboardInterrupt:
#         print(ConsoleColors.warning("\nDetención manual del bot"))
#     except Exception as e:
#         print(ConsoleColors.error(f"\nError en ejecución: {str(e)}"))
#     finally:
#         bot.cleanup()

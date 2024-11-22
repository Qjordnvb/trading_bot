
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional
from utils.console_colors import ConsoleColors
from core.market_analyzer import MarketAnalyzer
from core.meme_analyzer import MemeCoinAnalyzer
from models.data_classes import TradeRecommendation, TimingWindow
from models.enums import MarketTrend, TradingSignal, SignalStrength, EntryTiming



class BacktestSystem:
    def __init__(self, market_analyzer, meme_analyzer, client, config):
        self.market_analyzer = market_analyzer
        self.meme_analyzer = meme_analyzer
        self.client = client
        self.config = config
        self.results_cache = {}
        self.symbols_to_analyze = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']  # Añadir más pares
        self.analysis_period = 30
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'risk_reward_ratio': 0.0
        }
        self.current_symbol = None

    def run_backtest(self, symbol: str, start_time: str = None, end_time: str = None, initial_capital: float = 10000.0) -> Dict:
        """
        Ejecuta backtest completo para un símbolo
        """
        try:
            print(ConsoleColors.header(f"\n=== INICIANDO BACKTEST PARA {symbol} ==="))
            self.current_symbol = symbol

             # Obtener más datos históricos
            lookback_days = 90  # Aumentar de 30 a 90 días
            if not start_time:
                start_date = datetime.now() - timedelta(days=lookback_days)
                start_time = start_date.strftime("%Y-%m-%d")

            # Obtener fecha actual para validación de datos
            current_data = self.client.get_ticker_price(symbol)
            if not current_data:
                print(ConsoleColors.error(f"No se pudo obtener precio actual para {symbol}"))
                return None

            current_price = float(current_data['price'])
            print(ConsoleColors.info(f"Precio actual de {symbol}: ${current_price:,.2f}"))

            # Ajustar criterios de validación
            self.market_analyzer.strategy_params.update({
                "volume": {
                    "min_ratio": 1.2,  # Reducir de 1.5
                    "min_24h": 500000  # Reducir de 1000000
                },
                "risk_params": {
                    "max_risk_percent": 1.5,  # Reducir de 2.0
                    "risk_reward_min": 1.5    # Reducir de 2.0
                }
            })

            # Inicializar resultados
            results = {
                'trades': [],
                'metrics': {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'total_profit': 0,
                    'total_return': 0,
                    'max_drawdown': 0,
                    'profit_factor': 0,
                    'average_profit': 0,
                    'average_loss': 0,
                    'risk_metrics': {
                        'sharpe_ratio': 0,
                        'sortino_ratio': 0
                    }
                },
                'capital_curve': [initial_capital],
                'drawdown_curve': [0],
                'validation_scores': {},
                'test_period': {
                    'start': start_time,
                    'end': end_time,
                    'current_price': current_price
                }
            }

            # Obtener datos históricos con manejo de fechas actual/histórica
            historical_data = self._get_historical_data(symbol, start_time, end_time)
            if not historical_data:
                print(ConsoleColors.error(f"No se pudieron obtener datos válidos para {symbol}"))
                return None

            capital = initial_capital
            current_position = None
            max_capital = initial_capital
            max_drawdown = 0

            # Iterar sobre los datos históricos
            for i in range(len(historical_data)):
                candle = historical_data[i]

                try:
                    # Si no hay posición abierta, buscar señales de entrada
                    if not current_position:
                        signal = self._analyze_trading_opportunity(symbol, candle)
                        if signal and signal.signal == TradingSignal.BUY:
                            # Validar señal
                            if self._validate_signal(signal, historical_data[max(0, i-20):i+1]):
                                position_size = self._calculate_position_size(
                                    capital, signal.stop_loss, signal.entry_price
                                )
                                if position_size > 0:
                                    current_position = {
                                        'entry_price': signal.entry_price,
                                        'stop_loss': signal.stop_loss,
                                        'take_profit': signal.take_profit,
                                        'size': position_size,
                                        'entry_time': candle['timestamp']
                                    }

                    # Si hay posición abierta, verificar condiciones de salida
                    elif current_position:
                        exit_check = self._check_exit_conditions(
                            current_position,
                            candle,
                            self._analyze_trading_opportunity(symbol, candle)
                        )

                        if exit_check['should_exit']:
                            # Cerrar posición y calcular resultados
                            trade_result = self._close_position(
                                current_position,
                                exit_check['exit_price'],
                                candle['timestamp'],
                                exit_check['exit_reason']
                            )

                            # Actualizar capital y métricas
                            capital += trade_result['profit']
                            results['trades'].append(trade_result)
                            results['capital_curve'].append(capital)

                            # Actualizar drawdown
                            if capital > max_capital:
                                max_capital = capital
                            current_drawdown = (max_capital - capital) / max_capital
                            max_drawdown = max(max_drawdown, current_drawdown)
                            results['drawdown_curve'].append(current_drawdown)

                            current_position = None

                except Exception as e:
                    print(ConsoleColors.error(f"Error procesando vela: {str(e)}"))
                    continue

            # Calcular métricas finales
            results['metrics'] = self._calculate_performance_metrics(
                results['trades'],
                initial_capital,
                max_drawdown
            )

            # Validar resultados
            results['validation_scores'] = self._validate_backtest_results(results)

            # Imprimir resumen
            self._print_backtest_summary(results)

            # Guardar en caché
            self._cache_results(symbol, results)

            return results

        except Exception as e:
            print(ConsoleColors.error(f"Error en backtesting: {str(e)}"))
            import traceback
            print(ConsoleColors.error(traceback.format_exc()))
            return None

    def _get_historical_data(self, symbol: str, start_time: str, end_time: str) -> List[Dict]:
        """Obtiene datos históricos con validación mejorada"""
        try:
            print(ConsoleColors.info(f"\nObteniendo datos históricos para {symbol}"))

            # Obtener precio actual primero
            current_ticker = self.client.get_ticker_24h(symbol)
            if not current_ticker:
                print(ConsoleColors.error("No se pudo obtener precio actual"))
                return None

            current_price = float(current_ticker['lastPrice'])
            print(ConsoleColors.info(f"Precio actual: ${current_price:,.2f}"))

            # Obtener datos más recientes primero (últimas 24 horas)
            recent_data = self.client.get_klines(
                symbol=symbol,
                interval='1h',
                limit=24
            )

            if not recent_data:
                print(ConsoleColors.error("No se pudieron obtener datos recientes"))
                return None

            # Validar datos recientes
            latest_price = float(recent_data[-1]['close'])
            price_diff_pct = abs(latest_price - current_price) / current_price * 100

            if price_diff_pct > 1:  # Si hay más de 1% de diferencia
                print(ConsoleColors.warning(f"Diferencia significativa en precios recientes: {price_diff_pct:.2f}%"))
                return None

            # Obtener datos históricos solo si los datos recientes son válidos
            start_ts = int(datetime.strptime(start_time, "%Y-%m-%d").timestamp() * 1000)
            end_ts = int(datetime.strptime(end_time, "%Y-%m-%d").timestamp() * 1000)

            historical_data = self.client.get_klines(
                symbol=symbol,
                interval='1h',
                start_time=start_ts,
                end_time=end_ts
            )

            if not historical_data:
                print(ConsoleColors.error("No se pudieron obtener datos históricos"))
                return None

            # Combinar y validar datos
            all_data = []

            # Procesar datos históricos
            for candle in historical_data:
                if self._validate_candle(candle, current_price):
                    all_data.append(self._format_candle(candle))

            # Agregar datos recientes
            for candle in recent_data:
                if self._validate_candle(candle, current_price):
                    all_data.append(self._format_candle(candle))

            if not all_data:
                print(ConsoleColors.error("No hay datos válidos después del procesamiento"))
                return None

            print(ConsoleColors.success(f"Datos procesados: {len(all_data)} períodos"))
            return all_data

        except Exception as e:
            print(ConsoleColors.error(f"Error obteniendo datos históricos: {str(e)}"))
            return None

    def _validate_candle(self, candle: Dict, current_price: float) -> bool:
        """Valida que los datos de la vela sean coherentes"""
        try:
            close_price = float(candle['close'])
            # Permitir hasta 50% de diferencia con el precio actual para datos históricos
            if abs(close_price - current_price) / current_price > 0.5:
                return False
            return True
        except Exception:
            return False

    def _format_candle(self, candle: Dict) -> Dict:
        """Formatea los datos de la vela"""
        return {
            'timestamp': int(candle['timestamp']),
            'open': float(candle['open']),
            'high': float(candle['high']),
            'low': float(candle['low']),
            'close': float(candle['close']),
            'volume': float(candle['volume']),
            'trades': int(candle.get('trades', 0))
        }

    def _get_timeframe_ms(self, timeframe: str) -> int:
        """
        Convierte timeframe a milisegundos
        """
        multipliers = {
            '1h': 3600000,    # 1 hora
            '4h': 14400000,   # 4 horas
            '1d': 86400000    # 1 día
        }
        return multipliers.get(timeframe, 3600000)

    def _validate_time_continuity(self, timestamps: List[int]) -> bool:
        """
        Valida que no haya gaps significativos en los datos
        """
        if not timestamps:
            return False

        timestamps = sorted(timestamps)
        expected_interval = 3600000  # 1 hora en milisegundos
        max_gap = expected_interval * 2  # Permitir hasta 2 períodos de gap

        for i in range(1, len(timestamps)):
            if timestamps[i] - timestamps[i-1] > max_gap:
                return False

        return True

    def _process_historical_data(self, timeframes: Dict[str, List[Dict]]) -> List[Dict]:
       """
       Procesa y alinea datos de diferentes timeframes
       """
       try:
           # Validar datos de entrada
           if not all(timeframes.values()):
               print(ConsoleColors.warning("Faltan datos en algunos timeframes"))
               return []

           # Usar el timeframe más corto (1h) como base
           base_data = timeframes['1h']
           if not base_data:
               print(ConsoleColors.warning("No hay datos en timeframe base (1h)"))
               return []

           processed_data = []
           for candle in base_data:
               timestamp = int(candle['timestamp'])
               close_time = timestamp + 3600000  # 1 hora en milisegundos

               # Encontrar datos correspondientes en otros timeframes
               h4_data = next(
                   (c for c in timeframes['4h'] if timestamp >= int(c['timestamp']) and close_time <= int(c['close_time'])),
                   None
               )

               d1_data = next(
                   (c for c in timeframes['1d'] if timestamp >= int(c['timestamp']) and close_time <= int(c['close_time'])),
                   None
               )

               # Solo agregar si tenemos datos completos
               if h4_data and d1_data:
                   # Validar precios
                   base_close = float(candle['close'])
                   h4_close = float(h4_data['close'])
                   d1_close = float(d1_data['close'])

                   # Verificar coherencia de precios entre timeframes (máx 1% diferencia)
                   max_diff = max(
                       abs(base_close - h4_close) / base_close,
                       abs(base_close - d1_close) / base_close
                   )

                   if max_diff <= 0.01:  # 1% de diferencia máxima
                       processed_data.append({
                           **candle,
                           '4h': h4_data,
                           '1d': d1_data,
                           'timestamp': timestamp,
                           'close_time': close_time
                       })

           if not processed_data:
               print(ConsoleColors.warning("No se pudo procesar ningún dato histórico"))
               return []

           return processed_data

       except Exception as e:
           print(ConsoleColors.error(f"Error procesando datos históricos: {str(e)}"))
           print(f"Detalles del error: {str(e.__class__.__name__)}: {str(e)}")
           return []

    def _validate_signal(self, signal: TradeRecommendation, historical_data: List[Dict]) -> bool:
        """
        Valida señal usando múltiples criterios
        """
        try:
            print(ConsoleColors.info("\nValidando señal..."))
            validation_score = 0
            total_weight = 0

            # 1. Validación de tendencia (peso: 0.3)
            trend = self.market_analyzer._analyze_trend(historical_data)
            if signal.signal == TradingSignal.BUY and trend in [
                MarketTrend.STRONG_UPTREND,
                MarketTrend.UPTREND
            ]:
                validation_score += 0.3
                print(ConsoleColors.success("✓ Tendencia confirmada"))
            total_weight += 0.3

            # 2. Validación de volumen (peso: 0.2)
            volume_analysis = self.market_analyzer._analyze_volume(historical_data)
            if volume_analysis.get('is_significant'):
                validation_score += 0.2
                print(ConsoleColors.success("✓ Volumen significativo"))
            total_weight += 0.2

            # 3. Validación de momentum (peso: 0.2)
            momentum = self.market_analyzer._analyze_momentum(historical_data)
            if momentum.get('is_strong') and (
                (signal.signal == TradingSignal.BUY and momentum.get('is_positive')) or
                (signal.signal == TradingSignal.SELL and not momentum.get('is_positive'))
            ):
                validation_score += 0.2
                print(ConsoleColors.success("✓ Momentum confirmado"))
            total_weight += 0.2

            # 4. Validación de niveles (peso: 0.3)
            support_resistance = self.market_analyzer._calculate_support_resistance(historical_data)
            current_price = float(historical_data[-1]['close'])

            if signal.signal == TradingSignal.BUY:
                distance_to_support = abs(current_price - support_resistance['support']) / current_price
                if distance_to_support < 0.02:  # 2% del precio
                    validation_score += 0.3
                    print(ConsoleColors.success("✓ Precio cerca del soporte"))
            total_weight += 0.3

            # Calcular score final
            final_score = validation_score / total_weight if total_weight > 0 else 0
            print(ConsoleColors.info(f"Score de validación: {final_score:.2f}"))

            is_valid = final_score >= 0.7
            print(ConsoleColors.success("Señal validada") if is_valid else ConsoleColors.warning("Señal rechazada"))

            return is_valid

        except Exception as e:
            print(ConsoleColors.error(f"Error en validación de señal: {str(e)}"))
            return False

    def _calculate_position_size(self, capital: float, stop_loss: float, entry_price: float) -> float:
        """
        Calcula tamaño de posición basado en riesgo
        """
        try:
            risk_per_trade = capital * 0.02  # 2% riesgo por trade
            stop_loss_distance = abs(entry_price - stop_loss)

            if stop_loss_distance <= 0:
                return 0

            position_size = risk_per_trade / stop_loss_distance
            return min(position_size, capital / entry_price)  # No usar más del capital disponible

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando tamaño de posición: {str(e)}"))
            return 0

    def _check_exit_conditions(self, position: Dict, current_candle: Dict,
                             current_signal: TradeRecommendation) -> Dict:
        """
        Verifica condiciones de salida
        """
        try:
            current_price = float(current_candle['close'])
            should_exit = False
            exit_price = current_price
            exit_reason = ""

            # 1. Stop Loss
            if current_price <= position['stop_loss']:
                should_exit = True
                exit_price = position['stop_loss']
                exit_reason = "Stop Loss"

            # 2. Take Profit
            elif current_price >= position['take_profit']:
                should_exit = True
                exit_price = position['take_profit']
                exit_reason = "Take Profit"

            # 3. Señal de reversión
            elif current_signal and current_signal.signal == TradingSignal.SELL:
                should_exit = True
                exit_price = current_price
                exit_reason = "Señal de Reversión"

            return {
                'should_exit': should_exit,
                'exit_price': exit_price,
                'exit_reason': exit_reason
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error verificando condiciones de salida: {str(e)}"))
            return {'should_exit': False, 'exit_price': 0, 'exit_reason': ""}

    def _validate_backtest_results(self, results: Dict) -> Dict:
        """
        Valida los resultados del backtest
        """
        try:
            validation_scores = {
                'reliability_score': 0.0,
                'consistency_score': 0.0,
                'risk_score': 0.0,
                'overall_score': 0.0
            }

            # Calcular métricas básicas
            trades = results.get('trades', [])
            if not trades:
                return validation_scores

            # Calcular score de confiabilidad
            total_trades = len(trades)
            validation_scores['reliability_score'] = min(total_trades / 30, 1.0)  # Normalizar a 30 trades

            # Calcular score de consistencia
            if total_trades > 0:
                winning_trades = len([t for t in trades if t.get('profit', 0) > 0])
                win_rate = winning_trades / total_trades
                validation_scores['consistency_score'] = win_rate

            # Calcular score de riesgo
            max_drawdown = max(results.get('drawdown_curve', [0]))
            if max_drawdown > 0:
                risk_score = 1 - (max_drawdown / 100)  # Normalizar a porcentaje
                validation_scores['risk_score'] = max(0, min(risk_score, 1.0))

            # Calcular score general
            validation_scores['overall_score'] = (
                validation_scores['reliability_score'] * 0.3 +
                validation_scores['consistency_score'] * 0.4 +
                validation_scores['risk_score'] * 0.3
            )

            return validation_scores

        except Exception as e:
            print(ConsoleColors.error(f"Error validando resultados: {str(e)}"))
            return {
                'reliability_score': 0.0,
                'consistency_score': 0.0,
                'risk_score': 0.0,
                'overall_score': 0.0
            }



    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """
        Calcula el ratio de Sharpe
        """
        try:
            if not returns:
                return 0.0

            returns_array = np.array(returns)
            excess_returns = returns_array - (risk_free_rate / 252)  # Anualizar tasa libre de riesgo

            if len(excess_returns) < 2:
                return 0.0

            return (np.mean(excess_returns) / np.std(excess_returns, ddof=1)) * np.sqrt(252)

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando Sharpe ratio: {str(e)}"))
            return 0.0

    def _calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """
        Calcula el ratio de Sortino
        """
        try:
            if not returns:
                return 0.0

            returns_array = np.array(returns)
            excess_returns = returns_array - (risk_free_rate / 252)

            # Calcular desviación estándar de retornos negativos
            downside_returns = excess_returns[excess_returns < 0]
            if len(downside_returns) < 2:
                return 0.0

            downside_std = np.std(downside_returns, ddof=1)
            if downside_std == 0:
                return 0.0

            return (np.mean(excess_returns) / downside_std) * np.sqrt(252)

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando Sortino ratio: {str(e)}"))
            return 0.0

    def _analyze_trading_opportunity(self, symbol: str, candle: Dict) -> Optional[TradeRecommendation]:
        """
        Analiza una oportunidad de trading usando datos históricos
        """
        try:
            print(ConsoleColors.info(f"\nAnalizando oportunidad para {symbol}"))

            # 1. Validar datos de entrada
            if not self._validate_candle_data(candle):
                print(ConsoleColors.warning("Datos de vela inválidos"))
                return None

            # 2. Preparar datos para diferentes timeframes
            timeframes_data = self._prepare_timeframe_data(candle)
            if not timeframes_data:
                print(ConsoleColors.warning("No se pudieron preparar los timeframes"))
                return None

            # 3. Análisis técnico completo
            analysis = {
                # Análisis de tendencia en múltiples timeframes
                'trend_1h': self.market_analyzer._analyze_trend(timeframes_data['1h']),
                'trend_4h': self.market_analyzer._analyze_trend(timeframes_data['4h']),
                'trend_1d': self.market_analyzer._analyze_trend(timeframes_data['1d']),

                # Análisis de momentum
                'momentum_1h': self.market_analyzer._analyze_momentum(timeframes_data['1h']),
                'momentum_4h': self.market_analyzer._analyze_momentum(timeframes_data['4h']),

                # Análisis de volumen
                'volume': self.market_analyzer._analyze_volume(timeframes_data['1h']),

                # Volatilidad
                'volatility': self.market_analyzer._calculate_volatility(timeframes_data['1h']),

                # Soporte/Resistencia
                'levels': self.market_analyzer._calculate_support_resistance(timeframes_data['4h'])
            }

            # 4. Imprimir análisis detallado
            self._print_detailed_analysis(analysis)

            # 5. Evaluar condiciones de trading
            signal = self._evaluate_trading_conditions(analysis, candle)

            if signal:
                print(ConsoleColors.success("\n¡Señal de trading encontrada!"))
                return signal

            print(ConsoleColors.warning("\nNo se detectaron señales válidas"))
            return None

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando oportunidad: {str(e)}"))
            return None

    def _validate_candle_data(self, candle: Dict) -> bool:
        """
        Valida que los datos de la vela sean correctos
        """
        required_fields = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        if not all(field in candle for field in required_fields):
            return False

        # Validar que los precios sean coherentes
        if not (float(candle['low']) <= float(candle['close']) <= float(candle['high']) and
                float(candle['low']) <= float(candle['open']) <= float(candle['high'])):
            return False

        return True



    def _evaluate_trading_conditions(self, analysis: Dict, candle: Dict) -> Optional[TradeRecommendation]:
        try:
            current_price = float(candle['close'])

            # 1. Validar momentum y tendencia
            momentum_conditions = (
                analysis['momentum_1h']['is_positive'] and
                analysis['momentum_4h']['is_positive']
            )

            trend_conditions = (
                analysis['trend_1h'] in [MarketTrend.UPTREND, MarketTrend.STRONG_UPTREND] or
                analysis['trend_4h'] in [MarketTrend.UPTREND, MarketTrend.STRONG_UPTREND]
            )

            # 2. Validar volumen
            volume_conditions = (
                analysis['volume']['ratio'] > self.strategy_params["volume"]["min_ratio"] or
                analysis['volume']['is_increasing']
            )

            # 3. Validar niveles técnicos
            price_conditions = (
                current_price > analysis['levels']['support'] * 1.01 and
                current_price < analysis['levels']['resistance'] * 0.99
            )

            # Si se cumplen la mayoría de las condiciones
            conditions_met = sum([
                momentum_conditions,
                trend_conditions,
                volume_conditions,
                price_conditions
            ])

            if conditions_met >= 2:  # Reducir de 3 a 2 condiciones necesarias
                return TradeRecommendation(
                    signal=TradingSignal.BUY,
                    strength=SignalStrength.MODERATE if conditions_met == 2 else SignalStrength.STRONG,
                    reasons=self._generate_trade_reasons(analysis),
                    entry_price=current_price,
                    stop_loss=analysis['levels']['support'] * 0.995,
                    take_profit=analysis['levels']['resistance'] * 1.005
                )

            return None

        except Exception as e:
            print(ConsoleColors.error(f"Error evaluando condiciones: {str(e)}"))
            return None

    def _print_detailed_analysis(self, analysis: Dict):
        """
        Imprime análisis detallado
        """
        print(ConsoleColors.info("\nAnálisis de Tendencia:"))
        print(f"1H: {analysis['trend_1h']}")
        print(f"4H: {analysis['trend_4h']}")
        print(f"1D: {analysis['trend_1d']}")

        print(ConsoleColors.info("\nAnálisis de Momentum:"))
        print(f"1H: {'Positivo' if analysis['momentum_1h']['is_positive'] else 'Negativo'}")
        print(f"4H: {'Positivo' if analysis['momentum_4h']['is_positive'] else 'Negativo'}")

        print(ConsoleColors.info("\nAnálisis de Volumen:"))
        print(f"Ratio: {analysis['volume']['ratio']:.2f}x")
        print(f"Significativo: {'Sí' if analysis['volume']['is_significant'] else 'No'}")

        print(ConsoleColors.info("\nVolatilidad:"))
        print(f"{analysis['volatility']:.2%}")

        print(ConsoleColors.info("\nNiveles Clave:"))
        print(f"Soporte: ${analysis['levels']['support']:.2f}")
        print(f"Resistencia: ${analysis['levels']['resistance']:.2f}")

    def _close_position(self, position: Dict, exit_price: float,
                       exit_time: int, exit_reason: str) -> Dict:
        """
        Cierra posición y calcula resultados
        """
        try:
            entry_price = position['entry_price']
            position_size = position['size']

            profit = (exit_price - entry_price) * position_size
            profit_percentage = ((exit_price - entry_price) / entry_price) * 100

            return {
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position_size': position_size,
                'profit': profit,
                'profit_percentage': profit_percentage,
                'entry_time': position['entry_time'],
                'exit_time': exit_time,
                'duration': exit_time - position['entry_time'],
                'exit_reason': exit_reason
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error cerrando posición: {str(e)}"))
            return {}

    def _calculate_performance_metrics(self, trades: List[Dict], initial_capital: float, max_drawdown: float) -> Dict:
       """
       Calcula métricas completas de rendimiento
       """
       try:
           metrics = {
               'total_trades': len(trades),
               'winning_trades': len([t for t in trades if t['profit'] > 0]),
               'losing_trades': 0,
               'win_rate': 0,
               'total_profit': sum(t['profit'] for t in trades),
               'total_return': 0,
               'average_profit': 0,
               'average_loss': 0,
               'profit_factor': 0,
               'max_drawdown': max_drawdown * 100,
               'risk_metrics': {
                   'sharpe_ratio': 0,
                   'sortino_ratio': 0
               }
           }

           # Calcular métricas derivadas
           metrics['losing_trades'] = metrics['total_trades'] - metrics['winning_trades']

           if metrics['total_trades'] > 0:
               metrics['win_rate'] = (metrics['winning_trades'] / metrics['total_trades']) * 100
               metrics['total_return'] = (metrics['total_profit'] / initial_capital) * 100

               # Calcular promedios
               profits = [t['profit'] for t in trades if t['profit'] > 0]
               losses = [t['profit'] for t in trades if t['profit'] <= 0]

               metrics['average_profit'] = sum(profits) / len(profits) if profits else 0
               metrics['average_loss'] = sum(losses) / len(losses) if losses else 0

               # Calcular profit factor
               total_profits = sum(profits)
               total_losses = abs(sum(losses)) if losses else 1  # Evitar división por cero
               metrics['profit_factor'] = total_profits / total_losses

               # Calcular ratios
               returns = [t['profit_percentage'] for t in trades]
               metrics['risk_metrics']['sharpe_ratio'] = self._calculate_sharpe_ratio(returns)
               metrics['risk_metrics']['sortino_ratio'] = self._calculate_sortino_ratio(returns)

           return metrics

       except Exception as e:
           print(ConsoleColors.error(f"Error calculando métricas: {str(e)}"))
           return {
               'total_trades': 0,
               'winning_trades': 0,
               'losing_trades': 0,
               'win_rate': 0,
               'total_profit': 0,
               'total_return': 0,
               'max_drawdown': 0,
               'profit_factor': 0,
               'average_profit': 0,
               'average_loss': 0,
               'risk_metrics': {
                   'sharpe_ratio': 0,
                   'sortino_ratio': 0
               }
           }

    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """
        Calcula el ratio de Sharpe
        """
        try:
            if not returns:
                return 0.0

            # Convertir a retornos diarios si no lo están
            returns_array = np.array(returns)
            avg_return = np.mean(returns_array)
            std_dev = np.std(returns_array)

            if std_dev == 0:
                return 0.0

            # Usando tasa libre de riesgo de 2%
            risk_free_rate = 0.02 / 252  # Diaria

            return (avg_return - risk_free_rate) / std_dev * np.sqrt(252)

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando Sharpe ratio: {str(e)}"))
            return 0.0

    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """
        Calcula el ratio de Sortino (similar a Sharpe pero solo considera volatilidad negativa)
        """
        try:
            if not returns:
                return 0.0

            returns_array = np.array(returns)
            avg_return = np.mean(returns_array)

            # Calcular desviación estándar de retornos negativos
            negative_returns = returns_array[returns_array < 0]
            downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 0

            if downside_std == 0:
                return 0.0

            risk_free_rate = 0.02 / 252  # Tasa libre de riesgo diaria

            return (avg_return - risk_free_rate) / downside_std * np.sqrt(252)

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando Sortino ratio: {str(e)}"))
            return 0.0

    def _validate_backtest_results(self, results: Dict) -> Dict:
        """
        Valida los resultados del backtest
        """
        try:
            validation_scores = {
                'reliability_score': 0.0,
                'consistency_score': 0.0,
                'risk_score': 0.0,
                'overall_score': 0.0
            }

            # 1. Score de confiabilidad (basado en número de trades)
            total_trades = results['metrics']['total_trades']
            validation_scores['reliability_score'] = min(total_trades / 100, 1.0)

            # 2. Score de consistencia
            if total_trades > 0:
                profit_curve = self._analyze_profit_curve(results['capital_curve'])
                validation_scores['consistency_score'] = profit_curve['consistency_score']

            # 3. Score de riesgo
            risk_score = self._calculate_risk_score(results['metrics'])
            validation_scores['risk_score'] = risk_score

            # Score general
            validation_scores['overall_score'] = (
                validation_scores['reliability_score'] * 0.3 +
                validation_scores['consistency_score'] * 0.4 +
                validation_scores['risk_score'] * 0.3
            )

            return validation_scores

        except Exception as e:
            print(ConsoleColors.error(f"Error validando resultados: {str(e)}"))
            return {'reliability_score': 0, 'consistency_score': 0, 'risk_score': 0, 'overall_score': 0}

    def _analyze_profit_curve(self, capital_curve: List[float]) -> Dict:
        """
        Analiza la curva de capital para determinar consistencia
        """
        try:
            if len(capital_curve) < 2:
                return {'consistency_score': 0}

            # Calcular retornos diarios
            returns = np.diff(capital_curve) / capital_curve[:-1]

            # Calcular métricas de consistencia
            volatility = np.std(returns)
            trend = np.polyfit(range(len(returns)), returns, 1)[0]

            # Score basado en volatilidad y tendencia
            volatility_score = 1 - min(volatility * 10, 1)  # Menor volatilidad = mejor
            trend_score = min(max(trend * 100 + 0.5, 0), 1)  # Tendencia positiva = mejor

            consistency_score = (volatility_score * 0.6 + trend_score * 0.4)

            return {
                'consistency_score': consistency_score,
                'volatility': volatility,
                'trend': trend
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error analizando curva de beneficios: {str(e)}"))
            return {'consistency_score': 0, 'volatility': 0, 'trend': 0}

    def _prepare_timeframe_data(self, current_candle: Dict) -> Dict:
        """
        Prepara datos para diferentes timeframes
        """
        try:
            if not self.current_symbol:
                raise ValueError("Symbol not set")

            lookback_periods = {
                '1h': 50,
                '4h': 20,
                '1d': 10
            }

            timeframes_data = {}
            for tf, periods in lookback_periods.items():
                data = self._get_historical_candles(
                    symbol=self.current_symbol,
                    timestamp=current_candle['timestamp'],
                    timeframe=tf,
                    periods=periods
                )

                if len(data) >= periods:
                    timeframes_data[tf] = data
                else:
                    print(ConsoleColors.warning(f"Datos insuficientes para timeframe {tf}"))
                    return None

            return timeframes_data

        except Exception as e:
            print(ConsoleColors.error(f"Error preparando timeframes: {str(e)}"))
            return None

    def _get_historical_candles(self, symbol: str, timestamp: int, timeframe: str, periods: int) -> List[Dict]:
        try:
            ms_multiplier = {
                '1h': 3600000,
                '4h': 14400000,
                '1d': 86400000
            }

            start_ts = timestamp - (periods * ms_multiplier[timeframe])
            end_ts = timestamp

            print(f"Obteniendo datos para {symbol} en {timeframe}")
            candles = self.client.get_klines(
                symbol=symbol,
                interval=timeframe,
                limit=periods,
                start_time=start_ts,
                end_time=end_ts
            )

            if not candles:  # Add validation for empty response
                print(ConsoleColors.warning(f"No se obtuvieron datos para {symbol} en {timeframe}"))
                return []

            formatted_data = []
            for candle in candles:
                try:
                    # Validate candle data has enough elements
                    if len(candle) < 6:
                        continue

                    formatted_data.append({
                        'timestamp': int(candle['timestamp']) if isinstance(candle, dict) else int(candle[0]),
                        'open': float(candle['open']) if isinstance(candle, dict) else float(candle[1]),
                        'high': float(candle['high']) if isinstance(candle, dict) else float(candle[2]),
                        'low': float(candle['low']) if isinstance(candle, dict) else float(candle[3]),
                        'close': float(candle['close']) if isinstance(candle, dict) else float(candle[4]),
                        'volume': float(candle['volume']) if isinstance(candle, dict) else float(candle[5])
                    })
                except (IndexError, ValueError, TypeError) as e:
                    print(ConsoleColors.warning(f"Error formateando vela: {str(e)}"))
                    continue

            return formatted_data

        except Exception as e:
            print(ConsoleColors.error(f"Error obteniendo velas históricas: {str(e)}"))
            return []

    def _calculate_risk_score(self, metrics: Dict) -> float:
        """
        Calcula score de riesgo basado en métricas de trading
        """
        try:
            risk_score = 0.0

            # 1. Drawdown máximo (peso: 0.4)
            max_drawdown = metrics.get('max_drawdown', 100)
            drawdown_score = 1 - (max_drawdown / 100)
            risk_score += drawdown_score * 0.4

            # 2. Ratio de ganancia/pérdida (peso: 0.3)
            profit_factor = metrics.get('profit_factor', 0)
            pf_score = min(profit_factor / 3, 1.0)  # Normalizar a máximo de 3
            risk_score += pf_score * 0.3

            # 3. Win rate (peso: 0.3)
            win_rate = metrics.get('win_rate', 0)
            win_rate_score = win_rate / 100
            risk_score += win_rate_score * 0.3

            return risk_score

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando score de riesgo: {str(e)}"))
            return 0.0

    def _print_backtest_summary(self, results: Dict):
        """
        Imprime resumen del backtest
        """
        try:
            metrics = results['metrics']
            print(ConsoleColors.header("\n=== RESUMEN DE BACKTEST ==="))

            print(ConsoleColors.info("\nMétricas Generales:"))
            print(f"Total de operaciones: {metrics['total_trades']}")
            print(f"Operaciones ganadoras: {metrics['winning_trades']}")
            print(f"Operaciones perdedoras: {metrics['losing_trades']}")
            print(f"Win Rate: {metrics['win_rate']:.2f}%")

            print(ConsoleColors.info("\nMétricas de Rentabilidad:"))
            print(f"Beneficio total: ${metrics['total_profit']:.2f}")
            print(f"Retorno total: {metrics['total_return']:.2f}%")
            print(f"Beneficio promedio: ${metrics['average_profit']:.2f}")
            print(f"Factor de beneficio: {metrics['profit_factor']:.2f}")

            print(ConsoleColors.info("\nMétricas de Riesgo:"))
            print(f"Drawdown máximo: {metrics['max_drawdown']:.2f}%")
            print(f"Ratio Sharpe: {metrics['risk_metrics']['sharpe_ratio']:.2f}")
            print(f"Ratio Sortino: {metrics['risk_metrics']['sortino_ratio']:.2f}")

            validation = results['validation_scores']
            print(ConsoleColors.info("\nValidación de Resultados:"))
            print(f"Score de confiabilidad: {validation['reliability_score']:.2f}")
            print(f"Score de consistencia: {validation['consistency_score']:.2f}")
            print(f"Score de riesgo: {validation['risk_score']:.2f}")
            print(f"Score general: {validation['overall_score']:.2f}")

        except Exception as e:
            print(ConsoleColors.error(f"Error imprimiendo resumen: {str(e)}"))

    def _cache_results(self, symbol: str, results: Dict):
        """
        Guarda resultados en caché
        """
        self.results_cache[symbol] = {
            'results': results,
            'timestamp': datetime.now()
        }



import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta
import time
import logging
import json
import os
from decimal import Decimal, ROUND_DOWN
import math


class TradingBot:
    def __init__(self):
        self.api_key = "TU_API_KEY"
        self.api_secret = "TU_API_SECRET"

        # Inicializar el cliente de Binance
        self.client = Client(self.api_key, self.api_secret)

        # Configuración básica
        self.pairs = ["BTCUSDT", "ADAUSDT", "SHIBUSDT"]
        self.timeframe = "1h"
        self.stop_loss_percent = 0.05

        # Configuración de trading
        self.trade_amount = 20
        self.max_trades = 3
        self.min_profit = 1.5
        self.enable_trading = False

        # Crear carpeta de datos si no existe
        if not os.path.exists("data"):
            os.makedirs("data")

        # Configuración de logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filename="trading_bot.log",
        )
        self.logger = logging.getLogger(__name__)

    def verify_api_permissions(self):
        """
        Verifica los permisos de la API
        """
        try:
            # Prueba lectura básica
            status = self.client.get_account_status()
            if status:
                print("✅ Permiso de lectura: OK")

            # Prueba obtener balance
            balance = self.get_account_balance()
            if balance is not None:
                print("✅ Acceso a balance: OK")
                for b in balance:
                    if float(b["free"]) > 0:
                        print(f"   {b['asset']}: {b['free']}")
            else:
                print("❌ Acceso a balance: Error")

            # Prueba obtener precios
            price = self.get_current_price("BTCUSDT")
            if price:
                print(f"✅ Acceso a precios: OK (BTC: ${price})")
            else:
                print("❌ Acceso a precios: Error")

        except Exception as e:
            error_msg = str(e)
            if "API-key format invalid" in error_msg:
                print("❌ Error: Formato de API key inválido")
            elif "Invalid API-key, IP, or permissions for action" in error_msg:
                print("❌ Error: API key inválida o sin permisos suficientes")
            elif "Timestamp for this request" in error_msg:
                print("❌ Error: Problema de sincronización de tiempo")
            else:
                print(f"❌ Error desconocido: {error_msg}")

    def test_connection(self):
        """
        Prueba la conexión con Binance
        """
        try:
            self.client.ping()
            self.logger.info("Conexión exitosa con Binance")
            return True
        except Exception as e:
            self.logger.error(f"Error de conexión: {str(e)}")
            return False

    def get_account_balance(self):
        """
        Obtiene el balance de la cuenta
        """
        try:
            # Primero verificamos que la API tenga los permisos correctos
            account = self.client.get_account()
            if not account:
                self.logger.error("No se pudo obtener la información de la cuenta")
                return None

            # Obtenemos los balances
            balances = account["balances"]

            # Filtramos solo los balances con valor
            non_zero = [
                b for b in balances if float(b["free"]) > 0 or float(b["locked"]) > 0
            ]

            if not non_zero:
                self.logger.info("No se encontraron balances con valor")

            return non_zero

        except Exception as e:
            error_msg = str(e)
            if "API-key format invalid" in error_msg:
                self.logger.error("Error: API key con formato inválido")
            elif "Invalid API-key, IP, or permissions for action" in error_msg:
                self.logger.error("Error: API key inválida o sin permisos suficientes")
            elif "Timestamp for this request" in error_msg:
                self.logger.error("Error: Problema de sincronización de tiempo")
            else:
                self.logger.error(f"Error obteniendo balance: {error_msg}")
            return None

    def get_current_price(self, symbol):
        """
        Obtiene el precio actual de un par
        """
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker["price"])
        except Exception as e:
            self.logger.error(f"Error obteniendo precio para {symbol}: {str(e)}")
            return None

    def get_historical_data(self, symbol, interval="1h", limit=500):
        """
        Obtiene datos históricos de un par
        """
        try:
            klines = self.client.get_historical_klines(
                symbol=symbol, interval=interval, limit=limit
            )

            df = pd.DataFrame(
                klines,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "trades",
                    "taker_buy_base",
                    "taker_buy_quote",
                    "ignore",
                ],
            )

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            return df

        except Exception as e:
            self.logger.error(
                f"Error obteniendo datos históricos para {symbol}: {str(e)}"
            )
            return None

    def calculate_indicators(self, df):
        """
        Calcula los indicadores técnicos principales
        """
        try:
            prices = df["close"].values

            # RSI
            delta = np.diff(prices)
            gain = (delta * 0).copy()
            loss = (delta * 0).copy()
            gain[delta > 0] = delta[delta > 0]
            loss[delta < 0] = -delta[delta < 0]

            avg_gain = np.zeros_like(prices)
            avg_loss = np.zeros_like(prices)
            avg_gain[14] = np.mean(gain[:14])
            avg_loss[14] = np.mean(loss[:14])

            for i in range(15, len(prices)):
                avg_gain[i] = (avg_gain[i - 1] * 13 + gain[i - 1]) / 14
                avg_loss[i] = (avg_loss[i - 1] * 13 + loss[i - 1]) / 14

            rs = avg_gain[14:] / avg_loss[14:]
            rsi = 100 - (100 / (1 + rs))
            df["RSI"] = np.pad(rsi, (14, 0), mode="constant", constant_values=np.nan)

            # MACD
            exp1 = df["close"].ewm(span=12, adjust=False).mean()
            exp2 = df["close"].ewm(span=26, adjust=False).mean()
            df["MACD"] = exp1 - exp2
            df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
            df["MACD_Histogram"] = df["MACD"] - df["Signal_Line"]

            # Bollinger Bands
            df["BB_middle"] = df["close"].rolling(window=20).mean()
            bb_std = df["close"].rolling(window=20).std()
            df["BB_upper"] = df["BB_middle"] + (bb_std * 2)
            df["BB_lower"] = df["BB_middle"] - (bb_std * 2)

            # Moving Averages
            df["SMA_50"] = df["close"].rolling(window=50).mean()
            df["EMA_50"] = df["close"].ewm(span=50, adjust=False).mean()

            return df

        except Exception as e:
            self.logger.error(f"Error calculando indicadores: {str(e)}")
            return None

    def generate_signals(self, df):
        """
        Genera señales de trading basadas en los indicadores
        """
        try:
            signals = pd.DataFrame(index=df.index)
            signals["signal"] = 0

            # Señales RSI
            signals.loc[df["RSI"] < 30, "RSI_signal"] = 1
            signals.loc[df["RSI"] > 70, "RSI_signal"] = -1

            # Señales MACD
            signals.loc[df["MACD"] > df["Signal_Line"], "MACD_signal"] = 1
            signals.loc[df["MACD"] < df["Signal_Line"], "MACD_signal"] = -1

            # Señales Bollinger Bands
            signals.loc[df["close"] < df["BB_lower"], "BB_signal"] = 1
            signals.loc[df["close"] > df["BB_upper"], "BB_signal"] = -1

            # Señales Moving Averages
            signals.loc[df["SMA_50"] > df["EMA_50"], "MA_signal"] = 1
            signals.loc[df["SMA_50"] < df["EMA_50"], "MA_signal"] = -1

            # Señal combinada
            buy_conditions = (
                (signals["RSI_signal"] == 1)
                & (signals["MACD_signal"] == 1)
                & ((signals["BB_signal"] == 1) | (signals["MA_signal"] == 1))
            )
            sell_conditions = (
                (signals["RSI_signal"] == -1)
                & (signals["MACD_signal"] == -1)
                & ((signals["BB_signal"] == -1) | (signals["MA_signal"] == -1))
            )

            signals.loc[buy_conditions, "signal"] = 1
            signals.loc[sell_conditions, "signal"] = -1

            return signals

        except Exception as e:
            self.logger.error(f"Error generando señales: {str(e)}")
            return None

    def analyze_market(self, symbol):
        """
        Analiza el mercado y genera señales de trading
        """
        try:
            df = self.get_historical_data(symbol)
            if df is None:
                return None

            df = self.calculate_indicators(df)
            if df is None:
                return None

            signals = self.generate_signals(df)
            if signals is None:
                return None

            analysis = pd.concat([df, signals], axis=1)
            analysis.to_csv(f"data/{symbol.lower()}_analysis.csv")

            last_signal = signals["signal"].iloc[-1]
            current_price = self.get_current_price(symbol)

            return {
                "symbol": symbol,
                "current_price": current_price,
                "signal": last_signal,
                "RSI": df["RSI"].iloc[-1],
                "MACD": df["MACD"].iloc[-1],
                "BB_position": (df["close"].iloc[-1] - df["BB_lower"].iloc[-1])
                / (df["BB_upper"].iloc[-1] - df["BB_lower"].iloc[-1])
                * 100,
            }

        except Exception as e:
            self.logger.error(f"Error en análisis de mercado para {symbol}: {str(e)}")
            return None

    def get_symbol_info(self, symbol):
        """
        Obtiene información del par de trading
        """
        try:
            info = self.client.get_symbol_info(symbol)
            if info:
                return {
                    "min_qty": float(info["filters"][2]["minQty"]),
                    "step_size": float(info["filters"][2]["stepSize"]),
                    "min_notional": float(info["filters"][3]["minNotional"]),
                    "price_precision": info["quotePrecision"],
                    "quantity_precision": info["baseAssetPrecision"],
                }
            return None
        except Exception as e:
            self.logger.error(f"Error obteniendo info del par {symbol}: {str(e)}")
            return None

    def calculate_position_size(self, symbol, usdt_amount):
        """
        Calcula el tamaño correcto de la posición según las reglas del mercado
        """
        try:
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return None

            current_price = self.get_current_price(symbol)
            if not current_price:
                return None

            raw_quantity = usdt_amount / current_price
            step_size = symbol_info["step_size"]
            precision = int(round(-math.log10(step_size)))
            quantity = float(
                Decimal(str(raw_quantity)).quantize(
                    Decimal(str(step_size)), rounding=ROUND_DOWN
                )
            )

            if quantity < symbol_info["min_qty"]:
                self.logger.warning(f"Cantidad menor al mínimo permitido para {symbol}")
                return None

            if quantity * current_price < symbol_info["min_notional"]:
                self.logger.warning(
                    f"Valor total menor al mínimo permitido para {symbol}"
                )
                return None

            return quantity

        except Exception as e:
            self.logger.error(f"Error calculando tamaño de posición: {str(e)}")
            return None

    def place_market_order(self, symbol, side, quantity):
        """
        Coloca una orden de mercado
        """
        if not self.enable_trading:
            self.logger.warning("Trading real está desactivado")
            return None

        try:
            order = self.client.create_order(
                symbol=symbol, side=side, type="MARKET", quantity=quantity
            )

            self.logger.info(f"Orden ejecutada: {symbol} {side} {quantity}")
            return order

        except Exception as e:
            self.logger.error(f"Error colocando orden: {str(e)}")
            return None

    def execute_trade_strategy(self, symbol, analysis):
        """
        Ejecuta la estrategia de trading con verificaciones adicionales
        """
        try:
            # 1. Verificaciones previas
            if not self.verify_balance(symbol):
                print(f"Balance insuficiente para operar {symbol}")
                return None

            if self.check_open_orders(symbol):
                print(f"Ya hay órdenes abiertas para {symbol}")
                return None

            # 2. Verificar señal actual
            signal = analysis["signal"]
            if signal == 0:
                return None

            # 3. Calcular tamaño de la posición
            quantity = self.calculate_position_size(symbol, self.trade_amount)
            if not quantity:
                return None

            # 4. Obtener precio actual para cálculos
            current_price = self.get_current_price(symbol)

            # 5. Ejecutar orden según la señal
            if signal == 1:  # Señal de compra
                # Calcular stop loss y take profit
                stop_loss_price = current_price * (1 - self.stop_loss_percent)
                take_profit_price = current_price * (
                    1 + (self.stop_loss_percent * 1.5)
                )  # 1.5x el riesgo

                # Colocar orden de compra
                order = self.place_market_order(symbol, "BUY", quantity)
                if order:
                    # Colocar órdenes de stop loss y take profit
                    self.place_stop_loss_order(symbol, quantity, stop_loss_price)
                    self.place_take_profit_order(symbol, quantity, take_profit_price)
                    self.register_trade(symbol, "BUY", quantity, order["price"])

            elif signal == -1:  # Señal de venta
                order = self.place_market_order(symbol, "SELL", quantity)
                if order:
                    self.register_trade(symbol, "SELL", quantity, order["price"])

            return order

        except Exception as e:
            self.logger.error(f"Error ejecutando estrategia: {str(e)}")
            return None

    def register_trade(self, symbol, side, quantity, price):
        """
        Registra los trades ejecutados
        """
        try:
            trade_info = {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "timestamp": datetime.now().isoformat(),
                "value": quantity * price,
            }

            # Guardar en archivo CSV
            df = pd.DataFrame([trade_info])
            df.to_csv(
                f"data/trades_{symbol.lower()}.csv",
                mode="a",
                header=not os.path.exists(f"data/trades_{symbol.lower()}.csv"),
                index=False,
            )

        except Exception as e:
            self.logger.error(f"Error registrando trade: {str(e)}")


if __name__ == "__main__":
    bot = TradingBot()

    if bot.test_connection():
        print("\n=== Verificando configuración de API ===")
        bot.verify_api_permissions()
        print("\n=== Configuración del Bot ===")
        print(
            "Trading automático:", "ACTIVADO" if bot.enable_trading else "DESACTIVADO"
        )
        print(f"Pares configurados: {', '.join(bot.pairs)}")
        print(f"Cantidad por operación: {bot.trade_amount} USDT")
        print(f"Stop Loss: {bot.stop_loss_percent * 100}%")

        # Verificar balance
        balance = bot.get_account_balance()
        if balance:
            print("\nBalance inicial:")
            for b in balance:
                if float(b["free"]) > 0:
                    print(f"{b['asset']}: {b['free']}")
        else:
            print("\nNo se pudo obtener el balance.")
            respuesta = input("¿Deseas continuar en modo monitoreo solamente? (s/n): ")
            if respuesta.lower() != "s":
                print("Deteniendo el bot...")
                exit()

        # Bucle principal
        while True:
            try:
                for pair in bot.pairs:
                    analysis = bot.analyze_market(pair)
                    if analysis:
                        print(f"\n{'='*50}")
                        print(
                            f"Análisis para {pair} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        print(f"{'='*50}")
                        print(f"Precio actual: ${analysis['current_price']:.8f}")

                        signal_text = "MANTENER"
                        if analysis["signal"] == 1:
                            signal_text = "🟢 COMPRA"
                        elif analysis["signal"] == -1:
                            signal_text = "🔴 VENTA"
                        print(f"Señal: {signal_text}")

                        print(f"RSI: {analysis['RSI']:.2f}")
                        print(f"MACD: {analysis['MACD']:.8f}")
                        print(
                            f"Posición en Bandas de Bollinger: {analysis['BB_position']:.2f}%"
                        )

                        if bot.enable_trading:
                            print("\nEjecutando estrategia de trading...")
                            order = bot.execute_trade_strategy(pair, analysis)
                            if order:
                                print(f"Orden ejecutada: {order}")
                            else:
                                print("No se ejecutó ninguna orden")

                print(f"\nEsperando 60 segundos para el próximo análisis...")
                time.sleep(60)

            except Exception as e:
                bot.logger.error(f"Error en el ciclo principal: {str(e)}")
                print(f"Error en el ciclo principal: {str(e)}")
                print("Esperando 60 segundos antes de reintentar...")
                time.sleep(60)

    def monitor_positions(self):
        """
        Monitorea las posiciones abiertas
        """
        try:
            print("\nMonitoreando posiciones abiertas:")
            positions = self.get_open_positions()

            for pos in positions:
                current_price = self.get_current_price(pos["symbol"])
                entry_price = float(pos["entry_price"])
                pnl_percent = ((current_price - entry_price) / entry_price) * 100

                print(f"\n{pos['symbol']}:")
                print(f"Entrada: ${entry_price:.8f}")
                print(f"Actual: ${current_price:.8f}")
                print(f"P&L: {pnl_percent:.2f}%")

                # Verificar stop loss
                if pnl_percent <= -self.stop_loss_percent:
                    print("¡Stop Loss alcanzado! Cerrando posición...")
                    self.close_position(pos["symbol"], pos["quantity"])

        except Exception as e:
            self.logger.error(f"Error monitoreando posiciones: {str(e)}")

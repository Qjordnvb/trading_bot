# alerts/alert_manager.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import json
import time
import os
from .notifications import NotificationService
from utils.console_colors import ConsoleColors
from models.data_classes import TradeRecommendation, TimingWindow
from models.enums import TradingSignal, SignalStrength, EntryTiming


@dataclass
class Alert:
    symbol: str
    type: str
    condition: str
    target_price: float
    current_price: float
    message: str
    timestamp: datetime
    additional_info: Dict = field(default_factory=dict)
    is_triggered: bool = False


class AlertManager:
    def __init__(self, notifier: NotificationService):
        self.alerts: List[Alert] = []
        self.notifier = notifier
        self._initialize_alerts_file()
        self.load_alerts()
        self.last_alert_time = {}  # Track última alerta por símbolo
        self.MIN_ALERT_INTERVAL = 3600  # 1 hora entre alertas

    def _initialize_alerts_file(self):
        if not os.path.exists("alerts.json"):
            try:
                with open("alerts.json", "w") as f:
                    json.dump([], f)
            except Exception as e:
                print(
                    ConsoleColors.error(f"Error creando archivo de alertas: {str(e)}")
                )

    def add_price_alert(
        self,
        symbol: str,
        target_price: float,
        current_price: float,
        condition: str = "above",
        additional_info: Dict = None,
    ) -> None:
        message = f"{symbol} ha {'superado' if condition == 'above' else 'caído por debajo de'} {target_price:.8f}"
        self._add_alert(
            "price",
            symbol,
            target_price,
            current_price,
            condition,
            message,
            additional_info,
        )

    def add_support_resistance_alert(
        self,
        symbol: str,
        level: float,
        level_type: str,
        current_price: float,
        additional_info: Dict = None,
    ) -> None:
        message = f"{symbol} ha tocado el nivel de {level_type} en {level:.8f}"
        self._add_alert(
            level_type, symbol, level, current_price, "touch", message, additional_info
        )

    def _add_alert(
        self,
        alert_type: str,
        symbol: str,
        target_price: float,
        current_price: float,
        condition: str,
        message: str,
        additional_info: Dict = None,
    ) -> None:
        alert = Alert(
            symbol=symbol,
            type=alert_type,
            condition=condition,
            target_price=target_price,
            current_price=current_price,
            message=message,
            timestamp=datetime.now(),
            additional_info=additional_info or {},
        )
        self.alerts.append(alert)
        self.save_alerts()

    def check_alerts(self, market_data: Dict[str, Dict]) -> None:
        try:
            current_time = time.time()
            processed_alerts = set()

            for alert in self.alerts:
                if alert.is_triggered or alert.symbol in processed_alerts:
                    continue

                symbol = alert.symbol
                # Evitar alertas frecuentes del mismo símbolo
                if (
                    symbol in self.last_alert_time
                    and (current_time - self.last_alert_time[symbol])
                    < self.MIN_ALERT_INTERVAL
                ):
                    continue

                current_price = market_data.get(symbol, {}).get("price", 0)
                if not current_price:
                    continue

                if self._should_trigger_alert(alert, current_price):
                    signal = market_data[symbol].get("signal")
                    strength = market_data[symbol].get("strength")
                    entry_price = market_data[symbol].get("entry_price", 0)
                    stop_loss = market_data[symbol].get("stop_loss", 0)
                    take_profit = market_data[symbol].get("take_profit", 0)
                    stop_loss_percent = market_data[symbol].get("stop_loss_percent", 0)
                    take_profit_percent = market_data[symbol].get(
                        "take_profit_percent", 0
                    )

                    success = self._trigger_alert(
                        alert,
                        signal,
                        strength,
                        entry_price,
                        stop_loss,
                        take_profit,
                        stop_loss_percent,
                        take_profit_percent,
                    )

                    if success:
                        self.last_alert_time[symbol] = current_time
                        processed_alerts.add(symbol)
                        alert.is_triggered = True
                        self.save_alerts()

                    # Esperar entre alertas para no exceder límites de API
                    time.sleep(2)

        except Exception as e:
            print(f"Error en check_alerts: {str(e)}")

    def _should_trigger_alert(self, alert: Alert, current_price: float) -> bool:
        if alert.type == "price":
            return (
                current_price >= alert.target_price
                if alert.condition == "above"
                else current_price <= alert.target_price
            )
        elif alert.type in ["support", "resistance"]:
            return abs(current_price - alert.target_price) / alert.target_price <= 0.001
        return False

    def _trigger_alert(
        self,
        alert: Alert,
        signal,
        strength,
        entry_price,
        stop_loss,
        take_profit,
        stop_loss_percent,
        take_profit_percent,
    ) -> None:
        try:
            message = self._generate_alert_message(
                alert,
                signal,
                strength,
                entry_price,
                stop_loss,
                take_profit,
                stop_loss_percent,
                take_profit_percent,
            )
            if message:  # Solo enviar si el mensaje se generó correctamente
                self.notifier.send_message(message)
                alert.is_triggered = True
                self.save_alerts()
                print(ConsoleColors.success(f"Alerta enviada: {message}"))
        except Exception as e:
            print(ConsoleColors.error(f"Error enviando alerta: {str(e)}"))

    def _generate_alert_message(
        self,
        alert: Alert,
        signal: str,
        strength: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        stop_loss_percent: float,
        take_profit_percent: float,
    ) -> str:
        try:
            info = alert.additional_info

            # Iconos para mejorar la legibilidad
            signal_icon = "📶"  # Icono de señal
            strength_icon = "💪"  # Icono de fuerza
            alert_icon = "🚨"  # Icono de alerta
            level_icon = "📊"  # Icono de niveles de operación
            timing_icon = "🕒"  # Icono de análisis de timing
            market_icon = "📈"  # Icono de métricas de mercado
            entry_icon = "🔔"  # Icono de viñeta para razones
            bullet_icon = "•"  # Icono de viñeta para condiciones
            price_icon = "💲"  # Icono de precio
            recommendation_icon = "🔍"  # Icono de recomendación
            check_icon = "✅"  # Icono de validación
            cross_icon = "❌"  # Icono de advertencia/invalidación

            message = f"""==================================================
{alert_icon} === ANÁLISIS DE {alert.symbol} ===

"""

            # Recomendación de Trading
            recommendation = info.get("timing_recommendation", "N/A")
            message += f"""{recommendation_icon} Recomendación de Trading:
{signal_icon} Señal: {signal}
{strength_icon} Fuerza: {strength}
{entry_icon} Entrada: ${entry_price:.8f}
{cross_icon} Salida: ${stop_loss:.8f} ({stop_loss_percent:.2f}%)
{check_icon} Venta: ${take_profit:.8f} ({take_profit_percent:.2f}%)

"""

            if info.get("conditions"):
                message += f"\n{check_icon} Razones:"
                for reason in info["conditions"]:
                    message += f"\n  {bullet_icon} {reason}"

            # Niveles de Operación
            message += f"""
{level_icon} Niveles de Operación:
{price_icon} Precio actual: ${alert.current_price:.8f}

"""

            if all(k in info for k in ["target_price", "stop_loss", "take_profit"]):
                stop_loss_percent = (
                    (info["stop_loss"] - alert.current_price)
                    / alert.current_price
                    * 100
                )
                take_profit_percent = (
                    (info["take_profit"] - alert.current_price)
                    / alert.current_price
                    * 100
                )
                message += f"""
  {bullet_icon} Entrada: ${info['target_price']:.8f}
  {cross_icon} Stop Loss: ${info['stop_loss']:.8f} ({stop_loss_percent:.2f}%)
  {check_icon} Take Profit: ${info['take_profit']:.8f} ({take_profit_percent:.2f}%)

"""

            # Análisis de Timing
            message += f"""
{timing_icon} Análisis de Timing:
{recommendation_icon} Recomendación: {info.get('timing_recommendation', 'N/A')}
🕰️ Timeframe: {info.get('timeframe', 'N/A')}
⚖️ Confianza: {info.get('confidence', 0):.1f}%

"""

            # Condiciones
            if info.get("conditions"):
                message += f"\n{check_icon} Condiciones:"
                for condition in info.get("conditions", []):
                    message += f"\n  {bullet_icon} {condition}"

            # Métricas de mercado
            message += f"""
{market_icon} Métricas de Mercado:
{bullet_icon} Volumen 24h: ${info.get('volume_24h', 0):,.2f}
{bullet_icon} Variación 24h: {info.get('change_24h', 0):+.2f}%
{bullet_icon} Volatilidad: {info.get('volatility', 0):.1f}%

"""

            return message

        except Exception as e:
            print(ConsoleColors.error(f"Error generando mensaje de alerta: {str(e)}"))
            print(f"Info disponible: {info}")
            return None

    def save_alerts(self) -> None:
        try:
            alerts_data = [
                {
                    "symbol": alert.symbol,
                    "type": alert.type,
                    "condition": alert.condition,
                    "target_price": alert.target_price,
                    "current_price": alert.current_price,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "additional_info": alert.additional_info,
                    "is_triggered": alert.is_triggered,
                }
                for alert in self.alerts
            ]
            with open("alerts.json", "w") as f:
                json.dump(alerts_data, f, indent=4)
        except Exception as e:
            print(ConsoleColors.error(f"Error guardando alertas: {str(e)}"))

    def load_alerts(self) -> None:
        try:
            if not os.path.exists("alerts.json"):
                return

            with open("alerts.json", "r") as f:
                alerts_data = json.load(f)

            self.alerts = [
                Alert(
                    symbol=alert_data["symbol"],
                    type=alert_data["type"],
                    condition=alert_data["condition"],
                    target_price=alert_data["target_price"],
                    current_price=alert_data["current_price"],
                    message=alert_data["message"],
                    timestamp=datetime.fromisoformat(alert_data["timestamp"]),
                    additional_info=alert_data.get("additional_info", {}),
                    is_triggered=alert_data["is_triggered"],
                )
                for alert_data in alerts_data
            ]
        except json.JSONDecodeError:
            print(
                ConsoleColors.warning(
                    "Archivo de alertas vacío o corrupto. Creando nuevo archivo."
                )
            )
            self._initialize_alerts_file()
        except Exception as e:
            print(ConsoleColors.error(f"Error cargando alertas: {str(e)}"))

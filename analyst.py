import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import List, Dict, Tuple
from enum import Enum
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()
API_KEY = os.getenv("CMC_API_KEY")

if not API_KEY:
    raise ValueError("Error: No se encontró CMC_API_KEY en las variables de entorno")
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json
import time
from typing import List, Dict

from requests import Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json
from typing import List, Dict


class TradingSignal(Enum):
    BUY = "COMPRAR"
    SELL = "VENDER"
    HOLD = "MANTENER"


from requests import Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json
from typing import List, Dict


class MemeCoinAnalyzer:
    def __init__(self, cmc_api_key: str):
        self.session = Session()
        self.session.headers.update(
            {
                "Accepts": "application/json",
                "X-CMC_PRO_API_KEY": cmc_api_key,
            }
        )
        self.base_url = "https://pro-api.coinmarketcap.com/v1"
        self.market_analyzer = MarketAnalyzer(cmc_api_key)

        # Configuración de thresholds para análisis
        self.thresholds = {
            "min_market_cap": 1000000,  # $1M mínimo market cap
            "min_volume": 100000,  # $100K mínimo volumen
            "min_liquidity_ratio": 0.05,  # 5% mínimo ratio volumen/market cap
            "ideal_market_cap": {"min": 10000000, "max": 500000000},  # $10M  # $500M
            "momentum": {
                "strong_positive": 15,  # 15% para momentum fuerte
                "weak_positive": 5,  # 5% para momentum débil
                "strong_negative": -15,  # -15% para momentum negativo fuerte
                "oversold": -20,  # -20% para sobreventa
                "overbought": 30,  # 30% para sobrecompra
            },
        }

        # Keywords expandidos para identificar meme coins
        self.meme_keywords = {
            "primary": ["doge", "shib", "pepe", "floki", "meme", "inu"],
            "secondary": [
                "elon",
                "wojak",
                "chad",
                "cat",
                "kitty",
                "safe",
                "moon",
                "rocket",
                "baby",
                "pup",
                "pug",
                "shiba",
                "akita",
                "corgi",
                "moon",
            ],
            "exclude": [
                "chain",
                "swap",
                "protocol",
                "finance",
            ],  # Palabras para excluir falsos positivos
        }

    def get_meme_category_id(self) -> str:
        """Obtiene el ID de la categoría de memes"""
        try:
            url = f"{self.base_url}/cryptocurrency/categories"
            response = self.session.get(url)
            data = json.loads(response.text)

            if "data" in data:
                for category in data["data"]:
                    if "meme" in category["name"].lower():
                        return category["id"]

            print("No se encontró la categoría de memes")
            return None

        except Exception as e:
            print(f"Error obteniendo categoría de memes: {str(e)}")
            return None

    def get_meme_coins(self) -> List[Dict]:
        """Obtiene las meme coins usando un sistema de filtrado mejorado"""
        try:
            url = f"{self.base_url}/cryptocurrency/listings/latest"
            parameters = {
                "start": "1",
                "limit": "200",
                "convert": "USD",
                "sort": "volume_24h",
                "sort_dir": "desc",
            }

            print("Obteniendo listado de criptomonedas...")
            response = self.session.get(url, params=parameters)
            data = response.json()

            if "data" not in data:
                print("No se encontraron datos")
                return []

            meme_coins = []
            for coin in data["data"]:
                if self._is_meme_coin(coin) and self._meets_basic_criteria(coin):
                    meme_coins.append(coin)

            print(f"Se encontraron {len(meme_coins)} meme coins")
            return meme_coins

        except Exception as e:
            print(f"Error obteniendo meme coins: {str(e)}")
            return []

    def get_coin_info(self, symbol: str) -> Dict:
        """Obtiene información detallada de una criptomoneda"""
        try:
            url = f"{self.base_url}/cryptocurrency/quotes/latest"
            parameters = {"symbol": symbol, "convert": "USD"}

            response = self.session.get(url, params=parameters)
            data = response.json()

            if "data" in data and symbol in data["data"]:
                return data["data"][symbol]
            return None

        except Exception as e:
            print(f"Error obteniendo información para {symbol}: {str(e)}")
            return None

    def get_top_meme_coins(self) -> List[Dict]:
        try:
            meme_coins = self.get_meme_coins()
            if not meme_coins:
                return []

            print("\nAnalizando meme coins encontradas...")
            analyzed_coins = []

            try:
                market_metrics = self.market_analyzer.get_market_metrics()
                market_sentiment = self.market_analyzer.calculate_market_sentiment(
                    market_metrics
                )
            except Exception as e:
                print(f"Error obteniendo métricas de mercado: {str(e)}")
                market_sentiment = 0

            for coin in meme_coins:
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        coin_info = self.get_coin_info(coin["symbol"])
                        if not coin_info:
                            break

                        time.sleep(0.5)
                        exchanges = self.get_coin_exchanges(coin["id"])

                        score, reasons = self.analyze_metrics(coin_info)
                        trading_signal, signal_reasons = self.analyze_trading_signal(
                            coin_info, market_sentiment
                        )
                        quote = coin_info.get("quote", {}).get("USD", {})

                        if score > 0:
                            analyzed_coin = {
                                "id": coin["id"],
                                "symbol": coin_info["symbol"],
                                "name": coin_info["name"],
                                "score": score,
                                "reasons": reasons,
                                "signal": trading_signal.value,
                                "signal_reasons": signal_reasons,
                                "price": quote.get("price", 0),
                                "market_cap": quote.get("market_cap", 0),
                                "volume_24h": quote.get("volume_24h", 0),
                                "percent_change_24h": quote.get(
                                    "percent_change_24h", 0
                                ),
                                "percent_change_7d": quote.get("percent_change_7d", 0),
                                "rank": coin_info.get("cmc_rank", 0),
                                "exchanges": exchanges,
                            }
                            analyzed_coins.append(analyzed_coin)
                            break

                    except Exception as e:
                        if attempt == max_retries - 1:
                            print(
                                f"Error analizando {coin.get('symbol', 'Unknown')}: {str(e)}"
                            )
                        else:
                            time.sleep(1)
                        continue

            analyzed_coins.sort(key=lambda x: x["score"], reverse=True)
            top_coins = analyzed_coins[:10]

            print(
                f"\nAnálisis completado. Top {len(top_coins)} meme coins identificadas"
            )
            return top_coins

        except Exception as e:
            print(f"Error en get_top_meme_coins: {str(e)}")
            return []

    def _is_meme_coin(self, coin: Dict) -> bool:
        """Determina si una moneda es meme coin usando criterios mejorados"""
        name_lower = coin["name"].lower()
        symbol_lower = coin["symbol"].lower()

        # Verificar palabras de exclusión
        if any(
            word in name_lower or word in symbol_lower
            for word in self.meme_keywords["exclude"]
        ):
            return False

        # Verificar keywords primarios (mayor peso)
        primary_match = any(
            keyword in name_lower or keyword in symbol_lower
            for keyword in self.meme_keywords["primary"]
        )
        if primary_match:
            return True

        # Verificar keywords secundarios (requiere múltiples coincidencias)
        secondary_matches = sum(
            1
            for keyword in self.meme_keywords["secondary"]
            if keyword in name_lower or keyword in symbol_lower
        )
        return secondary_matches >= 2

    def _meets_basic_criteria(self, coin: Dict) -> bool:
        """Verifica si la moneda cumple con criterios básicos"""
        try:
            usd_data = coin["quote"]["USD"]
            market_cap = usd_data.get("market_cap", 0)
            volume_24h = usd_data.get("volume_24h", 0)

            return (
                market_cap >= self.thresholds["min_market_cap"]
                and volume_24h >= self.thresholds["min_volume"]
                and (volume_24h / market_cap if market_cap > 0 else 0)
                >= self.thresholds["min_liquidity_ratio"]
            )
        except Exception:
            return False

    def analyze_metrics(self, coin_data: Dict) -> Tuple[float, List[str]]:
        """Analiza las métricas de una moneda"""
        score = 0
        reasons = []
        quote = coin_data.get("quote", {}).get("USD", {})

        try:
            # Análisis de volumen y Market Cap
            volume_24h = quote.get("volume_24h", 0)
            market_cap = quote.get("market_cap", 0)

            # Puntuación por liquidez
            if market_cap and market_cap > 0:
                volume_to_mcap = volume_24h / market_cap
                liquidity_score = min(volume_to_mcap * 10, 5)
                score += liquidity_score
                if liquidity_score > 3:
                    reasons.append(f"Alta liquidez (V/MC: {volume_to_mcap*100:.1f}%)")

            # Análisis de cambios de precio
            price_changes = {
                "24h": quote.get("percent_change_24h", 0),
                "7d": quote.get("percent_change_7d", 0),
            }

            # Puntuación por momentum
            momentum_score = 0
            if price_changes["24h"] > 0:
                momentum_score += min(price_changes["24h"] / 20, 2)
            if price_changes["7d"] > 0:
                momentum_score += min(price_changes["7d"] / 20, 2)

            score += momentum_score
            if momentum_score > 2:
                reasons.append(
                    f"Fuerte momentum (+{price_changes['24h']:.1f}% 24h, +{price_changes['7d']:.1f}% 7d)"
                )

            # Análisis de Market Cap
            if (
                self.thresholds["ideal_market_cap"]["min"]
                <= market_cap
                <= self.thresholds["ideal_market_cap"]["max"]
            ):
                score += 3
                reasons.append("Market cap ideal para trading")
            elif market_cap > self.thresholds["ideal_market_cap"]["max"]:
                score += 1
                reasons.append("Gran capitalización")

            # Análisis de volumen
            if volume_24h > 1000000:  # > $1M
                score += 2
                reasons.append(f"Alto volumen (${volume_24h/1000000:.1f}M)")

            return score, reasons

        except Exception as e:
            print(f"Error analizando métricas: {str(e)}")
            return 0, ["Error en análisis"]

    def _analyze_liquidity(self, quote: Dict) -> Tuple[float, List[str]]:
        """Analiza la liquidez de la moneda"""
        score = 0
        reasons = []

        volume_24h = quote.get("volume_24h", 0)
        market_cap = quote.get("market_cap", 0)

        if market_cap and market_cap > 0:
            volume_to_mcap = volume_24h / market_cap
            if volume_to_mcap >= 0.5:
                score += 5
                reasons.append(
                    f"Liquidez excepcional (V/MC: {volume_to_mcap*100:.1f}%)"
                )
            elif volume_to_mcap >= 0.2:
                score += 3
                reasons.append(f"Alta liquidez (V/MC: {volume_to_mcap*100:.1f}%)")
            elif volume_to_mcap >= 0.1:
                score += 1
                reasons.append(f"Liquidez moderada (V/MC: {volume_to_mcap*100:.1f}%)")

        return score, reasons

    def _analyze_momentum(self, quote: Dict) -> Tuple[float, List[str]]:
        """Analiza el momentum de la moneda"""
        score = 0
        reasons = []

        change_24h = quote.get("percent_change_24h", 0)
        change_7d = quote.get("percent_change_7d", 0)

        momentum = (change_24h * 0.4) + (change_7d * 0.6)

        if momentum >= self.thresholds["momentum"]["strong_positive"]:
            score += 4
            reasons.append(f"Momentum muy fuerte (+{momentum:.1f}%)")
        elif momentum >= self.thresholds["momentum"]["weak_positive"]:
            score += 2
            reasons.append(f"Momentum positivo (+{momentum:.1f}%)")
        elif momentum <= self.thresholds["momentum"]["strong_negative"]:
            score -= 2
            reasons.append(f"Momentum negativo ({momentum:.1f}%)")

        return score, reasons

    def _analyze_market_cap(self, quote: Dict) -> Tuple[float, List[str]]:
        """Analiza el market cap de la moneda"""
        score = 0
        reasons = []

        market_cap = quote.get("market_cap", 0)

        if (
            self.thresholds["ideal_market_cap"]["min"]
            <= market_cap
            <= self.thresholds["ideal_market_cap"]["max"]
        ):
            score += 3
            reasons.append("Market cap ideal para trading")
        elif market_cap > self.thresholds["ideal_market_cap"]["max"]:
            score += 1
            reasons.append("Market cap establecido")

        return score, reasons

    def _analyze_volume(self, quote: Dict) -> Tuple[float, List[str]]:
        """Analiza el volumen de trading"""
        score = 0
        reasons = []

        volume_24h = quote.get("volume_24h", 0)

        if volume_24h > 10000000:
            score += 3
            reasons.append(f"Volumen muy alto (${volume_24h/1000000:.1f}M)")
        elif volume_24h > 1000000:
            score += 2
            reasons.append(f"Buen volumen (${volume_24h/1000000:.1f}M)")

        return score, reasons

    def analyze_trading_signal(
        self, coin_data: Dict, market_sentiment: float
    ) -> Tuple[TradingSignal, List[str]]:
        """Analiza y genera señales de trading"""
        signal_score = 0
        signal_reasons = []

        try:
            quote = coin_data.get("quote", {}).get("USD", {})

            # Métricas clave
            price_changes = {
                "24h": quote.get("percent_change_24h", 0),
                "7d": quote.get("percent_change_7d", 0),
            }
            volume_24h = quote.get("volume_24h", 0)
            market_cap = quote.get("market_cap", 0)

            # Análisis de volumen
            if market_cap > 0:
                volume_to_mcap = volume_24h / market_cap
                if volume_to_mcap > 0.3:
                    signal_score += 2
                    signal_reasons.append("Alto ratio volumen/market cap")

            # Análisis de momentum
            momentum = (price_changes["24h"] * 0.7) + (price_changes["7d"] * 0.3)
            if momentum > self.thresholds["momentum"]["strong_positive"]:
                signal_score += 2
                signal_reasons.append(f"Fuerte momentum positivo ({momentum:.1f}%)")
            elif momentum < self.thresholds["momentum"]["strong_negative"]:
                signal_score -= 2
                signal_reasons.append(f"Fuerte momentum negativo ({momentum:.1f}%)")

            # Influencia del sentimiento de mercado
            if market_sentiment > 0:
                signal_score += 1
                signal_reasons.append("Sentimiento de mercado positivo")
            elif market_sentiment < 0:
                signal_score -= 1
                signal_reasons.append("Sentimiento de mercado negativo")

            # Análisis de sobrecompra/sobreventa
            if price_changes["24h"] < self.thresholds["momentum"]["oversold"]:
                signal_score += 1
                signal_reasons.append("Posible sobreventa (24h)")
            elif price_changes["24h"] > self.thresholds["momentum"]["overbought"]:
                signal_score -= 1
                signal_reasons.append("Posible sobrecompra (24h)")

            # Determinar señal final
            if signal_score >= 2:
                return TradingSignal.BUY, signal_reasons
            elif signal_score <= -2:
                return TradingSignal.SELL, signal_reasons
            else:
                return TradingSignal.HOLD, signal_reasons

        except Exception as e:
            print(f"Error en análisis de señal: {str(e)}")
            return TradingSignal.HOLD, ["Error en análisis"]

    def _analyze_volume_signal(
        self, volume_24h: float, market_cap: float
    ) -> Tuple[float, List[str]]:
        """Analiza señales basadas en volumen"""
        score = 0
        reasons = []

        if market_cap > 0:
            volume_to_mcap = volume_24h / market_cap
            if volume_to_mcap > 0.5:
                score += 2
                reasons.append("Volumen excepcional relativo al market cap")
            elif volume_to_mcap > 0.3:
                score += 1
                reasons.append("Alto ratio volumen/market cap")
            elif volume_to_mcap < 0.05:
                score -= 1
                reasons.append("Bajo volumen relativo al market cap")

        return score, reasons

    def _analyze_momentum_signal(
        self, change_24h: float, change_7d: float
    ) -> Tuple[float, List[str]]:
        """Analiza señales basadas en momentum"""
        score = 0
        reasons = []

        momentum = (change_24h * 0.7) + (change_7d * 0.3)

        if momentum > 20:
            score += 2
            reasons.append(f"Momentum extremadamente fuerte (+{momentum:.1f}%)")
        elif momentum > 10:
            score += 1
            reasons.append(f"Momentum positivo sostenido (+{momentum:.1f}%)")
        elif momentum < -15:
            score -= 2
            reasons.append(f"Momentum negativo significativo ({momentum:.1f}%)")

        return score, reasons

    def _analyze_overbought(
        self, change_24h: float, change_7d: float
    ) -> Tuple[float, List[str]]:
        """Analiza condiciones de sobrecompra/sobreventa"""
        score = 0
        reasons = []

        # Análisis 24h
        if change_24h < -25:
            score += 1
            reasons.append("Posible sobreventa extrema (24h)")
        elif change_24h < -15:
            score += 0.5
            reasons.append("Posible sobreventa (24h)")
        elif change_24h > 40:
            score -= 1
            reasons.append("Posible sobrecompra extrema (24h)")
        elif change_24h > 25:
            score -= 0.5
            reasons.append("Posible sobrecompra (24h)")

        # Análisis 7d
        if change_7d < -40:
            score += 0.5
            reasons.append("Sobreventa potencial (7d)")
        elif change_7d > 100:
            score -= 0.5
            reasons.append("Sobrecompra potencial (7d)")

        return score, reasons

    def _analyze_market_sentiment(self, sentiment: float) -> Tuple[float, List[str]]:
        """Analiza el impacto del sentimiento de mercado"""
        score = 0
        reasons = []

        if sentiment > 15:
            score += 1
            reasons.append("Sentimiento de mercado muy positivo")
        elif sentiment > 5:
            score += 0.5
            reasons.append("Sentimiento de mercado positivo")
        elif sentiment < -15:
            score -= 1
            reasons.append("Sentimiento de mercado muy negativo")
        elif sentiment < -5:
            score -= 0.5
            reasons.append("Sentimiento de mercado negativo")

        return score, reasons

    def get_coin_exchanges(self, coin_id: int) -> List[Dict]:
        """Obtiene información de exchanges con mejor manejo de errores y validación"""
        try:
            url = "https://api.coinmarketcap.com/data-api/v3/cryptocurrency/market-pairs/latest"

            # Obtener slug de la moneda
            coin_info = self._get_coin_slug(coin_id)
            if not coin_info:
                return []

            parameters = {
                "slug": coin_info["slug"],
                "start": "1",
                "limit": "5",
                "category": "spot",
                "centerType": "all",
                "sort": "cmc_rank_advanced",
                "direction": "desc",
                "spotUntracked": "true",
            }

            headers = {"Accept": "application/json"}
            response = requests.get(url, params=parameters, headers=headers)

            if response.status_code != 200:
                print(f"Error en la API: {response.status_code}")
                return []

            data = response.json()

            if not self._validate_exchange_data(data):
                return []

            return self._process_exchange_data(data["data"]["marketPairs"][:5])

        except Exception as e:
            print(f"Error obteniendo exchanges: {str(e)}")
            return []

    def _get_coin_slug(self, coin_id: int) -> Dict:
        """Obtiene el slug de una moneda por su ID"""
        try:
            url = f"{self.base_url}/cryptocurrency/info"
            response = self.session.get(url, params={"id": str(coin_id)})

            if response.status_code != 200:
                print(
                    f"Error obteniendo información de la moneda: {response.status_code}"
                )
                return None

            data = response.json()
            if "data" not in data or str(coin_id) not in data["data"]:
                print(f"No se encontró información para el ID {coin_id}")
                return None

            return data["data"][str(coin_id)]

        except Exception as e:
            print(f"Error obteniendo slug: {str(e)}")
            return None

    def _validate_exchange_data(self, data: Dict) -> bool:
        """Valida la estructura de datos de exchanges"""
        return (
            isinstance(data, dict)
            and "data" in data
            and isinstance(data["data"], dict)
            and "marketPairs" in data["data"]
            and isinstance(data["data"]["marketPairs"], list)
            and len(data["data"]["marketPairs"]) > 0
        )

    def _process_exchange_data(self, market_pairs: List[Dict]) -> List[Dict]:
        """Procesa y formatea datos de exchanges"""
        exchanges = []
        for pair in market_pairs:
            try:
                exchange_info = {
                    "name": pair["exchangeName"],
                    "pair": f"{pair['baseSymbol']}/{pair['quoteSymbol']}",
                    "rank": pair["rank"],
                    "exchange_slug": pair["exchangeSlug"],
                    "volume_24h": self._get_pair_volume(pair),
                    "last_updated": pair.get("lastUpdated", ""),
                    "market_url": pair.get("marketUrl", ""),
                }
                exchanges.append(exchange_info)
            except Exception as e:
                print(f"Error procesando par de trading: {str(e)}")
                continue
        return exchanges

    def _get_pair_volume(self, pair: Dict) -> float:
        """Extrae el volumen 24h de un par de trading"""
        try:
            if isinstance(pair.get("quotes"), list):
                for quote in pair["quotes"]:
                    if quote.get("name") == "USD":
                        return quote.get("volume24h", 0)
            return pair.get("volumeUsd", 0)
        except Exception:
            return 0


class MarketAnalyzer:
    def __init__(self, cmc_api_key: str):
        self.cmc_api_key = cmc_api_key
        self.headers = {"X-CMC_PRO_API_KEY": cmc_api_key, "Accept": "application/json"}
        self.base_url = "https://pro-api.coinmarketcap.com/v1"
        # Valores históricos de volatilidad para comparación
        self.volatility_thresholds = {
            "low": 0.5,  # 50% del promedio histórico
            "medium": 1.0,  # En el promedio
            "high": 2.0,  # Doble del promedio
        }

    def get_market_metrics(self) -> Dict:
        """Obtiene métricas globales del mercado"""
        try:
            url = f"{self.base_url}/global-metrics/quotes/latest"
            response = requests.get(url, headers=self.headers)
            data = response.json()
            if "data" not in data:
                raise ValueError("No se encontraron datos de mercado")
            return data["data"]
        except Exception as e:
            print(f"Error obteniendo métricas de mercado: {str(e)}")
            return {}

    def calculate_market_sentiment(self, metrics: Dict) -> float:
        """Calcula el sentimiento general del mercado con factores adicionales"""
        try:
            btc_dominance = metrics.get("btc_dominance", 50)
            usd_data = metrics.get("quote", {}).get("USD", {})

            # Métricas básicas
            market_cap_change = usd_data.get(
                "total_market_cap_yesterday_percentage_change", 0
            )
            volume_change = usd_data.get(
                "total_volume_24h_yesterday_percentage_change", 0
            )

            # Nuevas métricas
            altcoin_dominance = 100 - btc_dominance
            market_cap = usd_data.get("total_market_cap", 0)
            volume_24h = usd_data.get("total_volume_24h", 0)

            # Cálculo de volatilidad
            volatility_score = self._calculate_volatility_score(
                market_cap_change, volume_change
            )

            # Cálculo de liquidez
            liquidity_score = self._calculate_liquidity_score(volume_24h, market_cap)

            # Análisis de dominancia
            dominance_score = self._analyze_market_dominance(
                btc_dominance, altcoin_dominance
            )

            # Ponderación de factores
            sentiment = (
                (0.25 * market_cap_change)  # Peso reducido
                + (0.20 * volume_change)  # Peso reducido
                + (0.20 * dominance_score)  # Nuevo factor
                + (0.20 * liquidity_score)  # Nuevo factor
                + (0.15 * volatility_score)  # Nuevo factor
            )

            return round(sentiment, 2)

        except Exception as e:
            print(f"Error calculando sentimiento del mercado: {str(e)}")
            return 0.0

    def _calculate_volatility_score(
        self, market_cap_change: float, volume_change: float
    ) -> float:
        """Calcula un score de volatilidad basado en cambios de mercado"""
        try:
            # Promedio ponderado de cambios
            weighted_change = abs(market_cap_change * 0.7 + volume_change * 0.3)

            # Normalizar según thresholds
            if weighted_change <= self.volatility_thresholds["low"]:
                return 50  # Mercado estable
            elif weighted_change <= self.volatility_thresholds["medium"]:
                return 25  # Volatilidad moderada
            else:
                return -25  # Alta volatilidad

        except Exception:
            return 0

    def _calculate_liquidity_score(self, volume_24h: float, market_cap: float) -> float:
        """Calcula un score de liquidez basado en el ratio volumen/market cap"""
        try:
            if market_cap == 0:
                return 0

            volume_mcap_ratio = volume_24h / market_cap

            # Normalizar ratio
            if volume_mcap_ratio >= 0.15:
                return 50  # Alta liquidez
            elif volume_mcap_ratio >= 0.05:
                return 25  # Liquidez media
            else:
                return -25  # Baja liquidez

        except Exception:
            return 0

    def _analyze_market_dominance(
        self, btc_dominance: float, altcoin_dominance: float
    ) -> float:
        """Analiza el balance entre BTC y altcoins"""
        try:
            # Balance ideal cercano a 50/50
            balance = 100 - abs(btc_dominance - altcoin_dominance)

            if balance >= 70:  # Mercado balanceado
                return 50
            elif balance >= 40:  # Ligero desbalance
                return 25
            else:  # Alto desbalance
                return -25

        except Exception:
            return 0


class TopCryptoAnalyzer:
    def __init__(self, cmc_api_key: str):
        self.market_analyzer = MarketAnalyzer(cmc_api_key)
        self.headers = {"X-CMC_PRO_API_KEY": cmc_api_key, "Accept": "application/json"}
        self.base_url = "https://pro-api.coinmarketcap.com/v1"

        # Configuración de thresholds para análisis técnico
        self.thresholds = {
            "rsi": {"oversold": 30, "overbought": 70},
            "momentum": {"strong_buy": 15, "buy": 5, "strong_sell": -15, "sell": -5},
            "volume": {"significant_increase": 20, "significant_decrease": -20},
            "volatility": {"high": 0.15, "low": 0.05},
        }

    def get_top_cryptos(self, limit: int = 5) -> List[Dict]:
        """Obtiene las principales criptomonedas con manejo de errores mejorado"""
        try:
            url = f"{self.base_url}/cryptocurrency/listings/latest"
            parameters = {
                "start": "1",
                "limit": str(limit),
                "convert": "USD",
                "sort": "market_cap",
                "sort_dir": "desc",
            }

            response = requests.get(url, headers=self.headers, params=parameters)
            data = response.json()

            if "status" in data and data["status"]["error_code"] != 0:
                print(f"Error en la API: {data['status']['error_message']}")
                return []

            return data.get("data", [])

        except Exception as e:
            print(f"Error obteniendo top cryptos: {str(e)}")
            return []

    def analyze_trading_signal(
        self, crypto_data: Dict, market_sentiment: float
    ) -> TradingSignal:
        """Análisis técnico mejorado para criptomonedas"""
        try:
            usd_data = crypto_data["quote"]["USD"]
            signal_score = 0

            # Análisis de precio
            price_analysis = self._analyze_price_action(usd_data)
            signal_score += price_analysis["score"]

            # Análisis de volumen
            volume_analysis = self._analyze_volume(usd_data)
            signal_score += volume_analysis["score"]

            # Análisis técnico
            technical_analysis = self._analyze_technical_indicators(usd_data)
            signal_score += technical_analysis["score"]

            # Influencia del sentimiento de mercado
            sentiment_analysis = self._analyze_market_sentiment(market_sentiment)
            signal_score += sentiment_analysis["score"]

            # Determinar señal final
            if signal_score >= 2:
                return TradingSignal.BUY
            elif signal_score <= -2:
                return TradingSignal.SELL
            return TradingSignal.HOLD

        except Exception as e:
            print(f"Error en análisis de trading: {str(e)}")
            return TradingSignal.HOLD

    def _analyze_price_action(self, usd_data: Dict) -> Dict:
        """Análisis detallado de acción del precio"""
        score = 0
        reasons = []

        try:
            price_change_24h = usd_data.get("percent_change_24h", 0)
            price_change_7d = usd_data.get("percent_change_7d", 0)
            price_change_30d = usd_data.get("percent_change_30d", 0)

            # Análisis de tendencia
            trend = (
                price_change_24h * 0.2 + price_change_7d * 0.3 + price_change_30d * 0.5
            )

            if trend > self.thresholds["momentum"]["strong_buy"]:
                score += 2
                reasons.append(f"Fuerte tendencia alcista ({trend:.1f}%)")
            elif trend > self.thresholds["momentum"]["buy"]:
                score += 1
                reasons.append(f"Tendencia alcista moderada ({trend:.1f}%)")
            elif trend < self.thresholds["momentum"]["strong_sell"]:
                score -= 2
                reasons.append(f"Fuerte tendencia bajista ({trend:.1f}%)")
            elif trend < self.thresholds["momentum"]["sell"]:
                score -= 1
                reasons.append(f"Tendencia bajista moderada ({trend:.1f}%)")

            return {"score": score, "reasons": reasons}

        except Exception as e:
            print(f"Error en análisis de precio: {str(e)}")
            return {"score": 0, "reasons": ["Error en análisis de precio"]}

    def _analyze_volume(self, usd_data: Dict) -> Dict:
        """Análisis detallado de volumen"""
        score = 0
        reasons = []

        try:
            volume_change_24h = usd_data.get("volume_change_24h", 0)
            market_cap = usd_data.get("market_cap", 0)
            volume_24h = usd_data.get("volume_24h", 0)

            # Análisis de cambio de volumen
            if volume_change_24h > self.thresholds["volume"]["significant_increase"]:
                score += 1
                reasons.append(
                    f"Aumento significativo de volumen (+{volume_change_24h:.1f}%)"
                )
            elif volume_change_24h < self.thresholds["volume"]["significant_decrease"]:
                score -= 1
                reasons.append(
                    f"Disminución significativa de volumen ({volume_change_24h:.1f}%)"
                )

            # Análisis de ratio volumen/market cap
            if market_cap > 0:
                volume_mcap_ratio = volume_24h / market_cap
                if volume_mcap_ratio > 0.2:
                    score += 1
                    reasons.append("Alto ratio volumen/market cap")
                elif volume_mcap_ratio < 0.05:
                    score -= 1
                    reasons.append("Bajo ratio volumen/market cap")

            return {"score": score, "reasons": reasons}

        except Exception as e:
            print(f"Error en análisis de volumen: {str(e)}")
            return {"score": 0, "reasons": ["Error en análisis de volumen"]}

    def _analyze_technical_indicators(self, usd_data: Dict) -> Dict:
        """Análisis de indicadores técnicos"""
        score = 0
        reasons = []

        try:
            price_change_24h = usd_data.get("percent_change_24h", 0)
            price_change_7d = usd_data.get("percent_change_7d", 0)

            # RSI simulado
            rsi = self._simulate_rsi(price_change_24h, price_change_7d)
            if rsi <= self.thresholds["rsi"]["oversold"]:
                score += 1
                reasons.append(f"RSI indica sobreventa ({rsi:.0f})")
            elif rsi >= self.thresholds["rsi"]["overbought"]:
                score -= 1
                reasons.append(f"RSI indica sobrecompra ({rsi:.0f})")

            # Volatilidad
            volatility = self._calculate_volatility(usd_data)
            if volatility > self.thresholds["volatility"]["high"]:
                reasons.append(f"Alta volatilidad ({volatility:.1%})")
            elif volatility < self.thresholds["volatility"]["low"]:
                reasons.append(f"Baja volatilidad ({volatility:.1%})")

            return {"score": score, "reasons": reasons}

        except Exception as e:
            print(f"Error en análisis técnico: {str(e)}")
            return {"score": 0, "reasons": ["Error en análisis técnico"]}

    def _analyze_market_sentiment(self, market_sentiment: float) -> Dict:
        """Análisis del sentimiento de mercado"""
        score = 0
        reasons = []

        if market_sentiment > 15:
            score += 1
            reasons.append("Sentimiento de mercado muy positivo")
        elif market_sentiment > 5:
            score += 0.5
            reasons.append("Sentimiento de mercado positivo")
        elif market_sentiment < -15:
            score -= 1
            reasons.append("Sentimiento de mercado muy negativo")
        elif market_sentiment < -5:
            score -= 0.5
            reasons.append("Sentimiento de mercado negativo")

        return {"score": score, "reasons": reasons}

    def _simulate_rsi(self, change_24h: float, change_7d: float) -> float:
        """RSI simulado mejorado"""
        try:
            weighted_change = change_24h * 0.7 + change_7d * 0.3
            if weighted_change == 0:
                return 50

            gain = max(weighted_change, 0)
            loss = abs(min(weighted_change, 0))

            if loss == 0:
                return 100

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            return max(0, min(100, rsi))

        except Exception:
            return 50

    def _calculate_volatility(self, usd_data: Dict) -> float:
        """Calcula la volatilidad basada en cambios de precio"""
        try:
            changes = [
                abs(usd_data.get("percent_change_24h", 0)),
                abs(usd_data.get("percent_change_7d", 0)),
                abs(usd_data.get("percent_change_30d", 0)),
            ]
            return statistics.mean(changes) / 100 if changes else 0
        except Exception:
            return 0

    def _normalize(self, value: float, min_val: float, max_val: float) -> float:
        """Normaliza un valor entre -1 y 1"""
        try:
            return max(min((value - min_val) / (max_val - min_val) * 2 - 1, 1), -1)
        except Exception:
            return 0


class PromissingProjectsAnalyzer:
    def __init__(self, cmc_api_key: str):
        self.headers = {"X-CMC_PRO_API_KEY": cmc_api_key, "Accept": "application/json"}
        self.base_url = "https://pro-api.coinmarketcap.com/v1"
        self.market_analyzer = MarketAnalyzer(cmc_api_key)

    def get_promising_projects(self) -> List[Dict]:
        """Identifica proyectos prometedores basados en múltiples métricas"""
        max_retries = 3
        retry_delay = 61  # 61 segundos para asegurar que el límite de tasa se resetee

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    print(
                        f"Reintentando obtener proyectos prometedores (intento {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(retry_delay)

                url = f"{self.base_url}/cryptocurrency/listings/latest"
                parameters = {
                    "start": "1",
                    "limit": "200",
                    "convert": "USD",
                    "sort": "market_cap",  # Cambiado para obtener los más relevantes primero
                    "sort_dir": "desc",
                }

                response = requests.get(url, headers=self.headers, params=parameters)
                response_data = response.json()

                if "status" in response_data:
                    if response_data["status"].get("error_code") == 0:  # Success
                        if "data" not in response_data:
                            print("No se encontraron datos en la respuesta de la API")
                            continue

                        market_metrics = self.market_analyzer.get_market_metrics()
                        market_sentiment = (
                            self.market_analyzer.calculate_market_sentiment(
                                market_metrics
                            )
                        )

                        analyzed_projects = []
                        projects_processed = 0

                        for project in response_data["data"]:
                            try:
                                # Solo analizar los primeros 100 proyectos para evitar límites
                                if projects_processed >= 100:
                                    break

                                if self._is_promising_project(project):
                                    score = self._calculate_project_score(project)
                                    if score > 0:
                                        trading_signal, signal_reasons = (
                                            self.analyze_trading_signal(
                                                project, market_sentiment
                                            )
                                        )
                                        analyzed_projects.append(
                                            {
                                                "project": project,
                                                "score": score,
                                                "analysis": self._get_project_analysis(
                                                    project
                                                ),
                                                "signal": trading_signal.value,
                                                "signal_reasons": signal_reasons,
                                            }
                                        )

                                projects_processed += 1

                            except Exception as e:
                                print(
                                    f"Error al analizar proyecto {project.get('symbol', 'Unknown')}: {str(e)}"
                                )
                                continue

                        sorted_projects = sorted(
                            analyzed_projects, key=lambda x: x["score"], reverse=True
                        )[:10]
                        print(
                            f"Análisis completado. Se encontraron {len(sorted_projects)} proyectos prometedores"
                        )
                        return sorted_projects

                    elif (
                        response_data["status"].get("error_code") == 1008
                    ):  # Rate limit exceeded
                        if attempt < max_retries - 1:
                            print(
                                "Límite de tasa excedido. Esperando para reintentar..."
                            )
                            continue
                        else:
                            print("Se alcanzó el límite de intentos por límite de tasa")
                            return []
                    else:
                        print(
                            f"Error en la API: {response_data['status'].get('error_message')}"
                        )
                        return []

            except Exception as e:
                print(f"Error inesperado en get_promising_projects: {str(e)}")
                if attempt < max_retries - 1:
                    continue
                return []

        return []

    def _is_promising_project(self, project: Dict) -> bool:
        """Determina si un proyecto es prometedor basado en criterios básicos"""
        try:
            usd_data = project["quote"]["USD"]
            market_cap = usd_data.get("market_cap", 0)
            volume_24h = usd_data.get("volume_24h", 0)

            # Criterios más estrictos para filtrar proyectos
            return (
                market_cap > 10000000  # Market cap > $10M
                and volume_24h > 1000000  # Volumen 24h > $1M
                and market_cap < 10000000000  # Market cap < $10B
                and volume_24h / market_cap > 0.01  # Ratio volumen/market cap > 1%
            )
        except Exception:
            return False

    def _calculate_project_score(self, project: Dict) -> float:
        """Calcula una puntuación para un proyecto basada en múltiples factores"""
        score = 0
        usd_data = project["quote"]["USD"]

        # Factor de crecimiento
        growth_score = (
            (usd_data["percent_change_24h"] * 0.2)
            + (usd_data["percent_change_7d"] * 0.3)
            + (usd_data["percent_change_30d"] * 0.5)
        ) / 100

        # Factor de volumen y liquidez
        volume_mcap_ratio = (
            usd_data["volume_24h"] / usd_data["market_cap"]
            if usd_data["market_cap"] > 0
            else 0
        )
        liquidity_score = min(volume_mcap_ratio * 10, 5)

        # Factor de madurez del mercado
        market_maturity = min(
            (
                project["circulating_supply"] / project["max_supply"]
                if project["max_supply"]
                else 0.5
            ),
            1,
        )

        # Combinar scores
        score = (growth_score * 0.4) + (liquidity_score * 0.3) + (market_maturity * 0.3)

        return score

    def _get_project_analysis(self, project: Dict) -> Dict:
        """Genera un análisis detallado del proyecto"""
        usd_data = project["quote"]["USD"]
        return {
            "fortalezas": self._analyze_strengths(project, usd_data),
            "riesgos": self._analyze_risks(project, usd_data),
            "recomendaciones": self._generate_recommendations(project, usd_data),
        }

    def _analyze_strengths(self, project: Dict, usd_data: Dict) -> List[str]:
        """Analiza las fortalezas del proyecto"""
        strengths = []
        if usd_data["market_cap"] > 1000000000:
            strengths.append("Alta capitalización de mercado")
        if usd_data["volume_24h"] > usd_data["market_cap"] * 0.1:
            strengths.append("Alto volumen de trading")
        if usd_data["percent_change_30d"] > 0:
            strengths.append("Tendencia alcista sostenida")
        return strengths

    def _analyze_risks(self, project: Dict, usd_data: Dict) -> List[str]:
        """Analiza los riesgos del proyecto"""
        risks = []
        if usd_data["volume_24h"] < usd_data["market_cap"] * 0.01:
            risks.append("Bajo volumen de trading")
        if usd_data["percent_change_7d"] < -20:
            risks.append("Alta volatilidad reciente")
        if (
            project["circulating_supply"] / project["max_supply"]
            if project["max_supply"]
            else 0 < 0.2
        ):
            risks.append("Baja distribución de tokens")
        return risks

    def _generate_recommendations(self, project: Dict, usd_data: Dict) -> List[str]:
        """Genera recomendaciones específicas para el proyecto"""
        recommendations = []
        if usd_data["market_cap"] < 100000000:
            recommendations.append(
                "Considerar como inversión de alto riesgo/alta recompensa"
            )
        if usd_data["percent_change_24h"] < -10:
            recommendations.append("Esperar estabilización antes de entrar")
        if usd_data["volume_24h"] > usd_data["market_cap"] * 0.2:
            recommendations.append("Buena liquidez para entradas y salidas")
        return recommendations

    def analyze_trading_signal(
        self, project: Dict, market_sentiment: float
    ) -> Tuple[TradingSignal, List[str]]:
        """Analiza y genera señales de trading para proyectos prometedores"""
        signal_reasons = []
        usd_data = project["quote"]["USD"]

        # Calcular métricas clave
        price_change_24h = usd_data.get("percent_change_24h", 0)
        price_change_7d = usd_data.get("percent_change_7d", 0)
        price_change_30d = usd_data.get("percent_change_30d", 0)
        volume_24h = usd_data.get("volume_24h", 0)
        market_cap = usd_data.get("market_cap", 0)

        signal_score = 0

        # Análisis técnico
        technical_trend = (
            (price_change_24h * 0.2)
            + (price_change_7d * 0.3)
            + (price_change_30d * 0.5)
        )

        if technical_trend > 10:
            signal_score += 2
            signal_reasons.append(
                f"Tendencia técnica positiva ({technical_trend:.1f}%)"
            )
        elif technical_trend < -10:
            signal_score -= 2
            signal_reasons.append(
                f"Tendencia técnica negativa ({technical_trend:.1f}%)"
            )

        # Análisis de volumen
        volume_mcap_ratio = volume_24h / market_cap if market_cap > 0 else 0
        if volume_mcap_ratio > 0.2:
            signal_score += 1
            signal_reasons.append("Alto volumen de trading")
        elif volume_mcap_ratio < 0.05:
            signal_score -= 1
            signal_reasons.append("Bajo volumen de trading")

        # Factores fundamentales
        if market_cap > 100000000 and volume_mcap_ratio > 0.1:
            signal_score += 1
            signal_reasons.append("Buena capitalización y liquidez")

        # Influencia del sentimiento de mercado
        if market_sentiment > 0:
            signal_score += 1
            signal_reasons.append("Sentimiento de mercado favorable")
        elif market_sentiment < 0:
            signal_score -= 1
            signal_reasons.append("Sentimiento de mercado desfavorable")

        # Determinar señal final
        if signal_score >= 3:
            return TradingSignal.BUY, signal_reasons
        elif signal_score <= -2:
            return TradingSignal.SELL, signal_reasons
        else:
            return TradingSignal.HOLD, signal_reasons


class CryptoMarketAnalyzer:
    def __init__(self, cmc_api_key: str):
        self.meme_coin_analyzer = MemeCoinAnalyzer(cmc_api_key)
        self.top_crypto_analyzer = TopCryptoAnalyzer(cmc_api_key)
        self.promising_analyzer = PromissingProjectsAnalyzer(cmc_api_key)
        self.market_analyzer = MarketAnalyzer(cmc_api_key)

    def generate_complete_analysis(self):
        """Genera un análisis completo del mercado"""
        try:
            result = {}

            # Obtener métricas globales del mercado
            print("Obteniendo métricas globales del mercado...")
            market_metrics = self.market_analyzer.get_market_metrics()
            if market_metrics:
                market_sentiment = self.market_analyzer.calculate_market_sentiment(
                    market_metrics
                )
                result["market_sentiment"] = market_sentiment
                print(f"Sentimiento del mercado calculado: {market_sentiment:.2f}")

            # Análisis de top criptos
            print("\nObteniendo top criptomonedas...")
            top_cryptos = self.top_crypto_analyzer.get_top_cryptos()
            if top_cryptos:
                top_crypto_analysis = []
                for crypto in top_cryptos:
                    signal = self.top_crypto_analyzer.analyze_trading_signal(
                        crypto, result.get("market_sentiment", 0)
                    )
                    top_crypto_analysis.append({"crypto": crypto, "signal": signal})
                result["top_cryptos"] = top_crypto_analysis
                print(
                    f"Análisis completado para {len(top_crypto_analysis)} criptomonedas"
                )

            # Obtener meme coins
            print("\nAnalizando meme coins...")
            top_meme_coins = self.meme_coin_analyzer.get_top_meme_coins()
            if top_meme_coins:
                result["meme_coins"] = top_meme_coins
                print(f"Análisis completado para {len(top_meme_coins)} meme coins")

            # Obtener proyectos prometedores
            print("\nAnalizando proyectos prometedores...")
            promising_projects = self.promising_analyzer.get_promising_projects()
            if promising_projects:
                result["promising_projects"] = promising_projects
                print(
                    f"Análisis completado para {len(promising_projects)} proyectos prometedores"
                )

            return result if result else None

        except Exception as e:
            print(f"\nError durante el análisis: {str(e)}")
            import traceback

            traceback.print_exc()
            return None


def print_complete_analysis(analysis: Dict):
    """Imprime un análisis completo del mercado"""
    if not analysis:
        print("No se pudo obtener el análisis completo")
        return

    print("\n=== ANÁLISIS COMPLETO DEL MERCADO CRYPTO ===")

    if "market_sentiment" in analysis:
        print(f"\nSentimiento del Mercado: {analysis['market_sentiment']:.2f}")

    # Imprimir TOP 5 Criptomonedas
    if "top_cryptos" in analysis:
        print("\n=== TOP 5 CRIPTOMONEDAS ===")
        for crypto_analysis in analysis["top_cryptos"]:
            crypto = crypto_analysis["crypto"]
            print(f"\n{crypto['name']} ({crypto['symbol']})")
            print(f"Precio: ${crypto['quote']['USD']['price']:.2f}")
            print(f"Market Cap: ${crypto['quote']['USD']['market_cap']:,.2f}")
            print(f"Cambio 24h: {crypto['quote']['USD']['percent_change_24h']:.2f}%")
            print(
                f"Señal: {crypto_analysis['signal'].value}"
            )  # Accedemos al value del enum

    # Imprimir TOP 10 Meme Coins
    if "meme_coins" in analysis:
        print("\n=== TOP 10 MEME COINS ===")
        for coin in analysis["meme_coins"]:
            print(f"\n{coin['name']} ({coin['symbol']})")
            print(f"Precio: ${coin['price']:.8f}")
            print(f"Puntuación: {coin['score']:.2f}")
            print(f"Señal: {coin['signal']}")
            print(f"Market Cap: ${coin['market_cap']:,.2f}")
            print(f"Volumen 24h: ${coin['volume_24h']:,.2f}")
            if coin.get("rank"):
                print(f"Rank: #{coin['rank']}")
            print("Cambios de precio:")
            print(f"  24h: {coin.get('percent_change_24h', 0):>7.2f}%")
            print(f"  7d:  {coin.get('percent_change_7d', 0):>7.2f}%")

            # Imprimir información de exchanges con manejo más seguro
            print("Top 5 Exchanges:")
            if coin.get("exchanges"):
                for exchange in coin["exchanges"]:
                    print(
                        f"  • {exchange.get('name', 'N/A')} ({exchange.get('exchange_slug', 'N/A')})"
                    )
                    print(f"    Par: {exchange.get('pair', 'N/A')}")
                    if exchange.get("rank"):
                        print(f"    Rank: #{exchange.get('rank')}")
                    if exchange.get("volume_24h"):
                        print(f"    Volumen 24h: ${exchange.get('volume_24h', 0):,.2f}")
                    if exchange.get("exchange_notice"):
                        print(f"    Aviso: {exchange.get('exchange_notice')}")
            else:
                print("  No se encontró información de exchanges")

            print("Razones para el score:")
            for reason in coin.get("reasons", []):
                print(f"  • {reason}")
            print("Razones para la señal:")
            for reason in coin.get("signal_reasons", []):
                print(f"  • {reason}")

    # Imprimir TOP 10 Proyectos Prometedores
    if "promising_projects" in analysis:
        print("\n=== TOP 10 PROYECTOS PROMETEDORES ===")
        for project in analysis["promising_projects"]:
            project_data = project["project"]
            print(f"\n{project_data['name']} ({project_data['symbol']})")
            print(f"Puntuación: {project['score']:.2f}")
            print(f"Señal: {project['signal']}")

            # Imprimir análisis detallado
            analysis_data = project["analysis"]
            if analysis_data.get("fortalezas"):
                print("Fortalezas:")
                for strength in analysis_data["fortalezas"]:
                    print(f"  • {strength}")

            if analysis_data.get("riesgos"):
                print("Riesgos:")
                for risk in analysis_data["riesgos"]:
                    print(f"  • {risk}")

            if analysis_data.get("recomendaciones"):
                print("Recomendaciones:")
                for rec in analysis_data["recomendaciones"]:
                    print(f"  • {rec}")

            if project.get("signal_reasons"):
                print("Razones para la señal:")
                for reason in project["signal_reasons"]:
                    print(f"  • {reason}")

            # Datos financieros si están disponibles
            if "quote" in project_data and "USD" in project_data["quote"]:
                usd_data = project_data["quote"]["USD"]
                print(f"\nDatos financieros:")
                print(f"  Precio: ${usd_data.get('price', 0):,.8f}")
                print(f"  Market Cap: ${usd_data.get('market_cap', 0):,.2f}")
                print(f"  Volumen 24h: ${usd_data.get('volume_24h', 0):,.2f}")
                print("  Cambios de precio:")
                print(f"    24h: {usd_data.get('percent_change_24h', 0):>7.2f}%")
                print(f"    7d:  {usd_data.get('percent_change_7d', 0):>7.2f}%")
                print(f"    30d: {usd_data.get('percent_change_30d', 0):>7.2f}%")


def main():
    try:
        print("Iniciando análisis del mercado crypto...")
        print(
            f"Usando API key: {API_KEY[:5]}..." if API_KEY else "API key no encontrada"
        )

        analyzer = CryptoMarketAnalyzer(API_KEY)
        analysis = analyzer.generate_complete_analysis()
        print_complete_analysis(analysis)

    except requests.exceptions.RequestException as e:
        print(f"Error de conexión con la API: {str(e)}")
    except ValueError as e:
        print(f"Error de configuración: {str(e)}")
    except Exception as e:
        print(f"Error inesperado: {str(e)}")


if __name__ == "__main__":
    main()

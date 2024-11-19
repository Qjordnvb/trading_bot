import requests
from typing import Dict, List, Optional
from datetime import datetime
from utils.console_colors import ConsoleColors
import time

class CoinMarketCapClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://pro-api.coinmarketcap.com/v1"
        self.headers = {
            'X-CMC_PRO_API_KEY': api_key,
            'Accept': 'application/json'
        }
        # Lista de símbolos considerados stablecoins
        self.stablecoins = {'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP', 'USDD'}
        self.last_request_time = 0
        self.min_request_interval = 1.1  # 1.1 segundos entre requests
        self.request_window = 60  # ventana de 60 segundos
        self.max_requests_per_minute = 30  # máximo de requests por minuto
        self.request_count = 0
        self.window_start_time = time.time()

    def get_market_opportunities(self, limit: int = 10, include_trending: bool = True) -> List[Dict]:
        """Obtiene oportunidades de mercado combinando top coins y trending"""
        try:
            opportunities = []

            # Obtener top coins por market cap
            top_coins = self.get_top_coins(limit)
            if top_coins:
                opportunities.extend([
                    coin for coin in top_coins
                    if not self._is_stablecoin(coin.get('symbol', ''))
                ])

            # Obtener monedas en tendencia si está habilitado
            if include_trending:
                trending_coins = self.get_trending_coins(limit)
                if trending_coins:
                    for coin in trending_coins:
                        if not self._is_stablecoin(coin.get('symbol', '')):
                            # Verificar si la moneda ya está en opportunities
                            if not any(existing['symbol'] == coin['symbol'] for existing in opportunities):
                                opportunities.append(coin)

            # Ordenar por market cap
            sorted_opportunities = sorted(
                opportunities,
                key=lambda x: x.get('market_cap', 0),
                reverse=True
            )

            # Aplicar filtros de volumen y market cap
            filtered_opportunities = [
                coin for coin in sorted_opportunities
                if (coin.get('volume_24h', 0) >= 500000 and
                    coin.get('market_cap', 0) >= 1000000)
            ]

            return filtered_opportunities[:limit]

        except Exception as e:
            print(ConsoleColors.error(f"Error obteniendo oportunidades de mercado: {str(e)}"))
            print(ConsoleColors.error(f"Detalles: {str(e.__class__.__name__)}: {str(e)}"))
            return []

    def get_market_metrics(self, symbol: str) -> Dict:
        """Obtiene métricas detalladas de mercado para un símbolo"""
        try:
            # Limpiar y validar el símbolo
            base_symbol = symbol.replace('USDT', '')
            if not base_symbol:
                return {}

            # Construir los parámetros de la petición
            params = {
                'symbol': base_symbol,
                'convert': 'USD'
            }

            # Realizar la petición
            response = self._make_request('cryptocurrency/quotes/latest', params)
            if not response or 'data' not in response:
                return {}

            # Extraer los datos
            data = response.get('data', {})
            if not data or base_symbol not in data:
                return {}

            # Procesar los datos de la moneda
            coin_data = data[base_symbol]
            if not isinstance(coin_data, dict):
                return {}

            quote_data = coin_data.get('quote', {}).get('USD', {})
            if not quote_data:
                return {}

            # Construir y retornar el resultado
            return {
                'market_cap': quote_data.get('market_cap', 0),
                'volume_24h': quote_data.get('volume_24h', 0),
                'percent_change_1h': quote_data.get('percent_change_1h', 0),
                'percent_change_24h': quote_data.get('percent_change_24h', 0),
                'percent_change_7d': quote_data.get('percent_change_7d', 0),
                'market_dominance': quote_data.get('market_cap_dominance', 0),
                'rank': coin_data.get('cmc_rank', 0),
                'last_updated': quote_data.get('last_updated', ''),
                'total_supply': coin_data.get('total_supply', 0),
                'circulating_supply': coin_data.get('circulating_supply', 0)
            }

        except Exception as e:
            print(ConsoleColors.error(f"Error procesando datos de CMC para {symbol}: {str(e)}"))
            return {}



    def get_top_coins(self, limit: int = 10) -> List[Dict]:
        """Obtiene las top criptomonedas por market cap con mejor manejo de errores"""
        try:
            params = {
                'limit': limit * 2,  # Pedir más para compensar filtros
                'convert': 'USD',
                'sort': 'market_cap',
                'sort_dir': 'desc'
            }

            response = self._make_request('cryptocurrency/listings/latest', params)
            if not response or 'data' not in response:
                return []

            data = response['data']
            coins = []

            for coin in data:
                try:
                    quote = coin.get('quote', {}).get('USD', {})
                    coins.append({
                        'symbol': coin.get('symbol', ''),
                        'name': coin.get('name', ''),
                        'price': quote.get('price', 0),
                        'volume_24h': quote.get('volume_24h', 0),
                        'market_cap': quote.get('market_cap', 0),
                        'percent_change_24h': quote.get('percent_change_24h', 0),
                        'rank': coin.get('cmc_rank', 0),
                        'type': 'top_coin'
                    })
                except Exception as e:
                    print(ConsoleColors.warning(f"Error procesando moneda: {str(e)}"))
                    continue

            # Filtrar monedas sin datos válidos
            valid_coins = [
                coin for coin in coins
                if coin['price'] > 0 and coin['market_cap'] > 0
            ]

            return valid_coins[:limit]

        except Exception as e:
            print(ConsoleColors.error(f"Error obteniendo top coins: {str(e)}"))
            return []



    def _is_stablecoin(self, symbol: str) -> bool:
        """Verifica si un símbolo corresponde a una stablecoin"""
        return symbol in self.stablecoins


    def get_trending_coins(self, limit: int = 10) -> List[Dict]:
        """Obtiene las criptomonedas en tendencia"""
        try:
            params = {
                'limit': limit * 2,  # Pedir más para compensar filtros
                'sort': 'percent_change_24h',
                'sort_dir': 'desc',
                'convert': 'USD'
            }

            response = self._make_request('cryptocurrency/listings/latest', params)
            if not response or 'data' not in response:
                return []

            trending = []
            for coin in response['data']:
                try:
                    quote_data = coin.get('quote', {}).get('USD', {})
                    if not quote_data:
                        continue

                    volume_24h = quote_data.get('volume_24h', 0)
                    if volume_24h >= 500000:  # Filtro de volumen básico
                        trending.append({
                            'symbol': coin.get('symbol', ''),
                            'name': coin.get('name', ''),
                            'price': quote_data.get('price', 0),
                            'volume_24h': volume_24h,
                            'market_cap': quote_data.get('market_cap', 0),
                            'percent_change_24h': quote_data.get('percent_change_24h', 0),
                            'rank': coin.get('cmc_rank', 0),
                            'type': 'trending'
                        })
                except Exception as e:
                    print(ConsoleColors.warning(f"Error procesando moneda en tendencia: {str(e)}"))
                    continue

            # Filtrar y ordenar por cambio de precio
            valid_trending = [
                coin for coin in trending
                if coin['price'] > 0 and coin['market_cap'] > 0
            ]

            sorted_trending = sorted(
                valid_trending,
                key=lambda x: abs(x['percent_change_24h']),
                reverse=True
            )

            return sorted_trending[:limit]

        except Exception as e:
            print(ConsoleColors.error(f"Error obteniendo trending coins: {str(e)}"))
            return []




    def get_fear_greed_index(self) -> Dict:
        """Obtiene el índice de miedo y codicia actual"""
        try:
            data = self._make_request('global-metrics/quotes/latest')

            # El índice se calcula basado en múltiples métricas
            market_metrics = data['quote']['USD']
            volatility = abs(market_metrics['btc_dominance_24h_percentage_change'])
            volume_change = market_metrics['total_volume_24h_yesterday_percentage_change']

            # Calcular índice
            fear_value = self._calculate_fear_greed(volatility, volume_change)

            return {
                'value': fear_value,
                'classification': self._get_fear_greed_classification(fear_value),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(ConsoleColors.error(f"Error obteniendo fear & greed index: {str(e)}"))
            return {}

    def get_trending_topics(self) -> List[Dict]:
        """Obtiene temas en tendencia relacionados con crypto"""
        try:
            data = self._make_request('cryptocurrency/trending/gainers-losers')

            topics = []
            for item in data:
                topics.append({
                    'symbol': item['symbol'],
                    'name': item['name'],
                    'percent_change_24h': item['quote']['USD']['percent_change_24h'],
                    'volume_change_24h': item['quote']['USD']['volume_change_24h'],
                    'trend_score': self._calculate_trend_score(item)
                })

            return topics
        except Exception as e:
            print(ConsoleColors.error(f"Error obteniendo trending topics: {str(e)}"))
            return []

    def _calculate_social_sentiment(self, data: Dict) -> float:
        """Calcula un score de sentimiento basado en métricas sociales"""
        try:
            twitter_score = min(data.get('twitter', {}).get('followers', 0) / 100000, 1)
            reddit_score = min(data.get('reddit', {}).get('subscribers', 0) / 50000, 1)
            active_users_score = min(data.get('reddit', {}).get('active_users', 0) / 1000, 1)

            return (twitter_score * 0.4 + reddit_score * 0.3 + active_users_score * 0.3)
        except Exception:
            return 0.5

    def _calculate_fear_greed(self, volatility: float, volume_change: float) -> int:
        """Calcula el índice de miedo y codicia"""
        try:
            # Normalizar volatilidad y cambio de volumen
            vol_score = max(min(50 - (volatility * 2), 100), 0)
            vol_change_score = max(min(50 + (volume_change / 2), 100), 0)

            # Combinar scores
            return int((vol_score + vol_change_score) / 2)
        except Exception:
            return 50

    def _get_fear_greed_classification(self, value: int) -> str:
        """Clasifica el valor del índice de miedo y codicia"""
        if value >= 75:
            return "Extreme Greed"
        elif value >= 55:
            return "Greed"
        elif value >= 45:
            return "Neutral"
        elif value >= 25:
            return "Fear"
        else:
            return "Extreme Fear"

    def _calculate_trend_score(self, item: Dict) -> float:
        """Calcula un score de tendencia para un tema"""
        try:
            price_change = abs(item['quote']['USD']['percent_change_24h'])
            volume_change = abs(item['quote']['USD']['volume_change_24h'])

            # Normalizar y combinar scores
            price_score = min(price_change / 100, 1)
            volume_score = min(volume_change / 200, 1)

            return (price_score * 0.7 + volume_score * 0.3)
        except Exception:
            return 0.0

    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Realiza requests a la API de CoinMarketCap con rate limiting"""
        try:
            current_time = time.time()

            # Resetear contadores si estamos en una nueva ventana
            if current_time - self.window_start_time >= self.request_window:
                self.request_count = 0
                self.window_start_time = current_time

            # Verificar límite de requests por minuto
            if self.request_count >= self.max_requests_per_minute:
                wait_time = self.window_start_time + self.request_window - current_time
                if wait_time > 0:
                    print(ConsoleColors.warning(f"Esperando {wait_time:.2f}s para respetar rate limit..."))
                    time.sleep(wait_time)
                    self.request_count = 0
                    self.window_start_time = time.time()

            # Esperar el intervalo mínimo entre requests
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last_request
                time.sleep(sleep_time)

            # Realizar request
            url = f"{self.base_url}/{endpoint}"
            response = requests.get(
                url,
                headers=self.headers,
                params=params,
                timeout=10
            )

            # Actualizar contadores
            self.last_request_time = time.time()
            self.request_count += 1

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                print(ConsoleColors.warning("Rate limit alcanzado, esperando..."))
                time.sleep(60)  # Esperar 1 minuto completo
                return self._make_request(endpoint, params)  # Reintentar
            else:
                print(ConsoleColors.error(f"Error {response.status_code} en request a CMC: {response.text}"))
                return {}

        except requests.exceptions.Timeout:
            print(ConsoleColors.error(f"Timeout en request a CMC para {endpoint}"))
            return {}
        except requests.exceptions.RequestException as e:
            print(ConsoleColors.error(f"Error en request a CMC: {str(e)}"))
            return {}
        except Exception as e:
            print(ConsoleColors.error(f"Error inesperado en request a CMC: {str(e)}"))
            return {}

    def get_new_listings(self, days: int = 7, limit: int = 20) -> List[Dict]:
        """Obtiene las últimas listings respetando rate limits"""
        try:
            # Solo hacer un request inicial para obtener datos básicos
            params = {
                'start': 1,
                'limit': min(limit * 2, 100),  # CMC tiene límite de 100
                'convert': 'USD',
                'sort': 'date_added',
                'sort_dir': 'desc',
                'cryptocurrency_type': 'all'  # incluir todos los tipos
            }

            response = self._make_request('cryptocurrency/listings/latest', params)
            if not response or 'data' not in response:
                print(ConsoleColors.warning("No se obtuvieron datos de nuevos listings de CMC"))
                return []

            # Filtrar por fecha y criterios básicos
            from datetime import datetime, timedelta
            cutoff_date = datetime.now() - timedelta(days=days)

            filtered_listings = []
            for coin in response.get('data', []):
                try:
                    date_added = datetime.strptime(coin['date_added'], '%Y-%m-%dT%H:%M:%S.%fZ')
                    if date_added < cutoff_date:
                        continue

                    quote = coin.get('quote', {}).get('USD', {})
                    if not quote:
                        continue

                    # Validar volumen y market cap mínimos
                    volume_24h = quote.get('volume_24h', 0)
                    market_cap = quote.get('market_cap', 0)

                    if volume_24h < 100000 or market_cap <= 0:  # Mínimo $100k volumen
                        continue

                    filtered_listings.append({
                        'id': coin.get('id'),
                        'symbol': coin.get('symbol', ''),
                        'name': coin.get('name', ''),
                        'date_added': date_added.strftime('%Y-%m-%d %H:%M:%S'),
                        'price': quote.get('price', 0),
                        'volume_24h': volume_24h,
                        'market_cap': market_cap,
                        'percent_change_24h': quote.get('percent_change_24h', 0),
                        'circulating_supply': coin.get('circulating_supply'),
                        'total_supply': coin.get('total_supply'),
                        'max_supply': coin.get('max_supply'),
                        'cmc_rank': coin.get('cmc_rank'),
                        'tags': coin.get('tags', [])
                    })

                except Exception as e:
                    print(ConsoleColors.warning(f"Error procesando listing {coin.get('symbol', 'Unknown')}: {str(e)}"))
                    continue

            # Ordenar por volumen y retornar
            return sorted(
                filtered_listings,
                key=lambda x: x['volume_24h'],
                reverse=True
            )[:limit]

        except Exception as e:
            print(ConsoleColors.error(f"Error obteniendo nuevos listings: {str(e)}"))
            return []

    def _get_trending_tokens(self) -> List[Dict]:
        """Obtiene tokens en tendencia"""
        try:
            response = self._make_request('cryptocurrency/trending/latest')
            if not response or 'data' not in response:
                return []

            return response['data']
        except Exception:
            return []

    def _get_latest_listings(self, limit: int) -> List[Dict]:
        """Obtiene últimas listings con parámetros optimizados"""
        params = {
            'start': 1,
            'limit': limit,
            'convert': 'USD',
            'sort': 'date_added',
            'sort_dir': 'desc',
            'aux': 'platform,tags,max_supply,circulating_supply,total_supply'
        }

        response = self._make_request('cryptocurrency/listings/latest', params)
        return response.get('data', [])

    def _get_token_metadata(self, symbol: str) -> Dict:
        """Obtiene metadata detallada del token"""
        params = {'symbol': symbol}
        response = self._make_request('cryptocurrency/info', params)
        return response.get('data', {}).get(symbol, {})

    def _get_community_data(self, symbol: str) -> Dict:
        """Obtiene datos de comunidad del token"""
        try:
            # Usar endpoint de community trending
            params = {'symbol': symbol}
            response = self._make_request('community/trending/token', params)
            if not response or 'data' not in response:
                return {}

            return response['data']
        except Exception:
            return {}

    def _get_trending_data(self, symbol: str) -> Dict:
        """Obtiene datos de tendencia"""
        try:
            # Combinar datos de diferentes endpoints de trending
            trending_topic = self._make_request('cryptocurrency/trending/gainers-losers', {'symbol': symbol})
            trending_token = self._make_request('community/trending/token', {'symbol': symbol})

            return {
                'topic_data': trending_topic.get('data', {}),
                'token_data': trending_token.get('data', {})
            }
        except Exception:
            return {}

    def _calculate_potential_score(self, token_data: Dict) -> float:
        """Calcula score de potencial mejorado"""
        score = 0.0
        try:
            # 1. Métricas de mercado (30%)
            market_score = self._calculate_market_score(token_data)
            score += market_score * 0.30

            # 2. Métricas de comunidad (25%)
            community_score = self._calculate_community_score(token_data)
            score += community_score * 0.25

            # 3. Métricas de tendencia (25%)
            trend_score = self._calculate_trend_score(token_data)
            score += trend_score * 0.25

            # 4. Métricas fundamentales (20%)
            fundamental_score = self._calculate_fundamental_score(token_data)
            score += fundamental_score * 0.20

            return score * 10  # Convertir a escala 0-10

        except Exception as e:
            print(ConsoleColors.error(f"Error calculando potential score: {str(e)}"))
            return 0.0

    def _calculate_market_score(self, data: Dict) -> float:
        score = 0.0
        try:
            # Market cap bajo (bueno para x100)
            mcap = data.get('market_cap', 0)
            if mcap < 1000000: score += 0.4  # <$1M
            elif mcap < 5000000: score += 0.3  # <$5M
            elif mcap < 10000000: score += 0.2  # <$10M

            # Volumen significativo
            volume = data.get('volume_24h', 0)
            if volume > 1000000: score += 0.3
            elif volume > 500000: score += 0.2

            # Holders
            if data.get('metadata', {}).get('holder_count', 0) > 1000:
                score += 0.3

        except Exception:
            pass
        return min(score, 1.0)

    def _calculate_community_score(self, data: Dict) -> float:
        score = 0.0
        try:
            community = data.get('community', {})

            # Social media presence
            if community.get('twitter_followers', 0) > 10000: score += 0.3
            if community.get('telegram_members', 0) > 5000: score += 0.3
            if community.get('reddit_subscribers', 0) > 1000: score += 0.2

            # Engagement
            if community.get('social_engagement', 0) > 0.7: score += 0.2

        except Exception:
            pass
        return min(score, 1.0)

    def _calculate_trend_score(self, data: Dict) -> float:
        score = 0.0
        try:
            trending = data.get('trending', {})

            # Trending topic score
            if trending.get('topic_data', {}).get('is_trending', False):
                score += 0.4

            # Trending token score
            if trending.get('token_data', {}).get('is_trending', False):
                score += 0.4

            # Momentum
            price_change = data.get('quote', {}).get('USD', {}).get('percent_change_24h', 0)
            if price_change > 50: score += 0.2

        except Exception:
            pass
        return min(score, 1.0)

    def _calculate_fundamental_score(self, data: Dict) -> float:
        score = 0.0
        try:
            metadata = data.get('metadata', {})

            # Verificar utilidad/caso de uso
            if metadata.get('category') in ['defi', 'gaming', 'web3', 'ai']:
                score += 0.3

            # Verificar equipo y desarrollo
            if metadata.get('team_visible', False): score += 0.2
            if metadata.get('github_activity', False): score += 0.2

            # Verificar seguridad
            if metadata.get('is_audited', False): score += 0.3

        except Exception:
            pass
        return min(score, 1.0)

    def _combine_token_data(self, trending: List[Dict], listings: List[Dict]) -> List[Dict]:
        """Combina y deduplica datos de tokens"""
        combined = {token['symbol']: token for token in listings}

        # Añadir tokens trending si no están ya en listings
        for token in trending:
            if token['symbol'] not in combined:
                combined[token['symbol']] = token

        return list(combined.values())

    def get_social_stats(self, symbol: str) -> Dict:
        """Obtiene estadísticas sociales usando metadata de CMC"""
        try:
            params = {'symbol': symbol}
            data = self._make_request('cryptocurrency/info', params)

            if not data or 'data' not in data:
                return {}

            coin_data = data['data'].get(symbol, {})
            if not coin_data:
                return {}

            # Extraer métricas sociales disponibles
            urls = coin_data.get('urls', {})
            return {
                'twitter_followers': len(urls.get('twitter', [])),
                'telegram_members': len(urls.get('chat', [])),
                'reddit_subscribers': len(urls.get('reddit', [])),
                'github_repos': len(urls.get('source_code', [])),
                'website_url': urls.get('website', [None])[0],
                'explorer_url': urls.get('explorer', [None])[0],
                'technical_doc': urls.get('technical_doc', [None])[0],
                'social_score': self._calculate_social_presence(urls)
            }
        except Exception as e:
            print(ConsoleColors.error(f"Error obteniendo datos sociales para {symbol}: {str(e)}"))
            return {}

    def _calculate_social_presence(self, urls: Dict) -> float:
        """Calcula un score basado en la presencia en redes sociales"""
        score = 0.0
        weights = {
            'twitter': 0.3,
            'chat': 0.2,
            'reddit': 0.2,
            'source_code': 0.15,
            'technical_doc': 0.15
        }

        for platform, weight in weights.items():
            if platform in urls and urls[platform]:
                score += weight

        return score



    def _calculate_listing_risk(self, coin: Dict, quote: Dict) -> float:
        """Calcula un score de riesgo para nuevos listings (0-1, menor es mejor)"""
        try:
            risk_score = 0.0

            # Factor de volumen (0-0.3)
            volume = quote.get('volume_24h', 0)
            volume_score = min(volume / 1000000, 1.0) * 0.3  # Normalizar a $1M
            risk_score += 0.3 - volume_score

            # Factor de market cap (0-0.2)
            market_cap = quote.get('market_cap', 0)
            mcap_score = min(market_cap / 10000000, 1.0) * 0.2  # Normalizar a $10M
            risk_score += 0.2 - mcap_score

            # Factor de plataforma (0-0.2)
            platform = coin.get('platform', {}).get('name', '').lower()
            trusted_platforms = {'binance', 'ethereum', 'bsc', 'polygon'}
            if platform in trusted_platforms:
                risk_score += 0.0
            else:
                risk_score += 0.2

            # Factor de volatilidad (0-0.3)
            percent_change = abs(quote.get('percent_change_24h', 0))
            if percent_change > 100:  # Más de 100% de cambio es muy arriesgado
                risk_score += 0.3
            elif percent_change > 50:  # 50-100% es moderadamente arriesgado
                risk_score += 0.2
            elif percent_change > 20:  # 20-50% es poco arriesgado
                risk_score += 0.1

            return min(1.0, risk_score)

        except Exception:
            return 1.0  # Máximo riesgo en caso de error

    def get_supported_symbols(self) -> List[str]:
        """Obtiene lista de símbolos soportados en CMC"""
        try:
            response = self._make_request('cryptocurrency/map')
            if not response or 'data' not in response:
                return []

            return [coin['symbol'] for coin in response['data']]
        except Exception as e:
            print(ConsoleColors.error(f"Error obteniendo símbolos de CMC: {str(e)}"))
            return []

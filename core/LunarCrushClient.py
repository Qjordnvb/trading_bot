import requests
from typing import Dict, List
from config import config
from nltk.sentiment import SentimentIntensityAnalyzer

class LunarCrushClient:
    def __init__(self):
        self.base_url = "https://api.lunarcrush.com/v2"
        self.headers = {
            "accept": "application/json"
            "bearer": "yhwbuf5ij1sxuzbmoq8efd6ny8m3nwvxwes9c61u8"
        }
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    # ... (métodos anteriores)

    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        return sentiment_scores

    def get_coin_sentiment(self, coin_symbol: str) -> Dict:
        endpoint = f"/coins/{coin_symbol}/v1"
        response = self._make_request(endpoint)
        data = response.get("data", {})

        sentiment_data = {}
        if data:
            sentiment_data = {
                "sentiment_score": data.get("sentiment_score", 0),
                "sentiment_text": data.get("sentiment_text", ""),
                "news_sentiment": self._analyze_sentiment(data.get("news_sentiment_text", "")),
                "social_sentiment": self._analyze_sentiment(data.get("social_sentiment_text", ""))
            }

        return sentiment_data

    def get_category_sentiment(self, category_id: str) -> Dict:
        endpoint = f"/category/{category_id}/v1"
        response = self._make_request(endpoint)
        data = response.get("data", {})

        sentiment_data = {}
        if data:
            sentiment_data = {
                "sentiment_score": data.get("sentiment_score", 0),
                "sentiment_text": data.get("sentiment_text", ""),
                "news_sentiment": self._analyze_sentiment(data.get("news_sentiment_text", "")),
                "social_sentiment": self._analyze_sentiment(data.get("social_sentiment_text", ""))
            }

        return sentiment_data

    def get_topic_sentiment(self, topic_id: str) -> Dict:
        endpoint = f"/topic/{topic_id}/v1"
        response = self._make_request(endpoint)
        data = response.get("data", {})

        sentiment_data = {}
        if data:
            sentiment_data = {
                "sentiment_score": data.get("sentiment_score", 0),
                "sentiment_text": data.get("sentiment_text", ""),
                "news_sentiment": self._analyze_sentiment(data.get("news_sentiment_text", "")),
                "social_sentiment": self._analyze_sentiment(data.get("social_sentiment_text", ""))
            }

        return sentiment_data

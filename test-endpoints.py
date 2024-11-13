# test-endpoints.py

from requests import Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json
from dotenv import load_dotenv
import os


def test_api_endpoints(api_key: str):
    session = Session()
    session.headers.update(
        {
            "Accepts": "application/json",
            "X-CMC_PRO_API_KEY": api_key,
        }
    )
    base_url = "https://pro-api.coinmarketcap.com/v1"

    endpoints = [
        "/cryptocurrency/listings/latest",
        "/cryptocurrency/quotes/latest",
        "/cryptocurrency/categories",
        "/cryptocurrency/category",
        "/global-metrics/quotes/latest",
    ]

    print("Probando endpoints de la API...")
    print(f"API Key: {api_key[:5]}...")

    for endpoint in endpoints:
        try:
            url = base_url + endpoint
            params = {"limit": "1"}

            # Ajustes específicos para ciertos endpoints
            if endpoint == "/cryptocurrency/quotes/latest":
                params["symbol"] = "BTC"
            elif endpoint == "/cryptocurrency/category":
                params["id"] = "1"

            print(f"\nProbando {endpoint}")
            print(f"URL: {url}")
            print(f"Parámetros: {params}")

            response = session.get(url, params=params)
            data = response.json()

            if "status" in data:
                if data["status"]["error_code"] == 0:
                    print(f"✓ {endpoint}: OK")
                else:
                    print(f"✗ {endpoint}: {data['status']['error_message']}")
            else:
                print(f"✗ {endpoint}: Respuesta sin status")
                print("Respuesta:", data)
        except Exception as e:
            print(f"✗ {endpoint}: Error - {str(e)}")


if __name__ == "__main__":
    load_dotenv()
    API_KEY = os.getenv("CMC_API_KEY")

    if not API_KEY:
        print("Error: No se encontró la API key en las variables de entorno")
    else:
        test_api_endpoints(API_KEY)

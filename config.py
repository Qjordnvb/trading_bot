# config.py
from dotenv import load_dotenv
import os

# Cargar variables de entorno desde el archivo .env
load_dotenv()

class Config:
    CMC_API_KEY = os.getenv('CMC_API_KEY')
    # Aquí puedes agregar más API keys en el futuro
    GATE_IO_API_KEY = os.getenv('GATE_IO_API_KEY')
    GATE_IO_API_SECRET = os.getenv('GATE_IO_API_SECRET')

# main.py
from config import Config
from crypto_market_analyzer import CryptoMarketAnalyzer

def main():
    # Inicializar el analizador con la API key
    analyzer = CryptoMarketAnalyzer(Config.CMC_API_KEY)
    analysis = analyzer.generate_complete_analysis()
    print_complete_analysis(analysis)

if __name__ == "__main__":
    main()

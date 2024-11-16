# utils/console_colors.py
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

class ConsoleColors:
    @staticmethod
    def header(text: str) -> str:
        return f"{Fore.CYAN}{Style.BRIGHT}{text}{Style.RESET_ALL}"

    @staticmethod
    def success(text: str) -> str:
        return f"{Fore.GREEN}{text}{Style.RESET_ALL}"

    @staticmethod
    def error(text: str) -> str:
        return f"{Fore.RED}{text}{Style.RESET_ALL}"

    @staticmethod
    def warning(text: str) -> str:
        return f"{Fore.YELLOW}{text}{Style.RESET_ALL}"

    @staticmethod
    def info(text: str) -> str:
        return f"{Fore.BLUE}{text}{Style.RESET_ALL}"

    @staticmethod
    def highlight(text: str) -> str:
        return f"{Fore.MAGENTA}{text}{Style.RESET_ALL}"

    @staticmethod
    def price_change(change: float) -> str:
        return f"{Fore.GREEN if change >= 0 else Fore.RED}{change:+.2f}%{Style.RESET_ALL}"

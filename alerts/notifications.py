# alerts/notifications.py
from abc import ABC, abstractmethod
from twilio.rest import Client

class NotificationService(ABC):
    @abstractmethod
    def send_message(self, message: str) -> bool:
        pass

class WhatsAppNotifier(NotificationService):
    def __init__(self, account_sid: str, auth_token: str, from_number: str, to_number: str):
        self.client = Client(account_sid, auth_token)
        self.from_number = from_number
        self.to_number = to_number

    def send_message(self, message: str) -> bool:
        try:
            self.client.messages.create(
                body=message,
                from_=f'whatsapp:{self.from_number}',
                to=f'whatsapp:{self.to_number}'
            )
            return True
        except Exception as e:
            print(f"Error enviando mensaje de WhatsApp: {str(e)}")
            return False

class MockNotifier(NotificationService):
    """Notificador de prueba para desarrollo"""
    def send_message(self, message: str) -> bool:
        print(f"\n[MOCK NOTIFICATION]\n{message}\n")
        return True

from abc import ABC, abstractmethod


class BaseMQ(ABC):
    @abstractmethod
    def send(self, message: str):
        pass
    
    @abstractmethod
    def recv(self):
        pass
    
    @abstractmethod
    def start(self):
        pass
    
    @abstractmethod
    def stop(self):
        pass

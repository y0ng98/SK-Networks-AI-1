from abc import ABC, abstractmethod


class LangchainService(ABC):
    @abstractmethod
    def regWithLangchain(self, userSendMessage):
        pass
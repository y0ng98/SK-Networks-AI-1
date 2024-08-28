from abc import ABC, abstractmethod


class LanguageModelRepository(ABC):
    @abstractmethod
    def preprocessForCreateUniqueCharacter(self, text):
        pass

    @abstractmethod
    def preprocessForCreateTextIndex(self, text, chartToIndex):
        pass

    @abstractmethod
    def createDataSet(self, text, textAsIndex):
        pass

    @abstractmethod
    def trainModel(self, sequenceList, characterList):
        pass

    @abstractmethod
    def requestToReadShakespeareModel(self):
        pass

    @abstractmethod
    def convertTextToTensor(self, userInputText, charToIndex):
        pass

    @abstractmethod
    def generateText(self, loadedModel, inputTensor, indexToChar):
        pass
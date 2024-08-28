from sequence_analysis.repository.sequence_analysis_repository_impl import SequenceAnalysisRepositoryImpl
from sequence_analysis.service.sequence_analysis_service import SequenceAnalysisService


class SequenceAnalysisServiceImpl(SequenceAnalysisService):

    def __init__(self):
        self.__sequenceAnalysisRepository = SequenceAnalysisRepositoryImpl()

    def predictNextSequence(self, userSendMessage):
        print(f"service -> predictNextSequence(): userSendMessage: {userSendMessage}")

        tokenizer = self.__sequenceAnalysisRepository.createTokenizer(userSendMessage)
        totalWords, inputSequences = self.__sequenceAnalysisRepository.extractSequence(tokenizer, userSendMessage)
        maxSequenceLength = max([len(x) for x in inputSequences])

        paddedInputSequences = self.__sequenceAnalysisRepository.paddingSequence(inputSequences, maxSequenceLength)
        X, y = self.__sequenceAnalysisRepository.separateInputAndOutputSequences(paddedInputSequences, totalWords)
        traindModel = self.__sequenceAnalysisRepository.trainSequence(totalWords, maxSequenceLength, X, y)

        generatedText = self.__sequenceAnalysisRepository.generateText(
            "오늘도", 2, traindModel, maxSequenceLength, tokenizer)

        return generatedText
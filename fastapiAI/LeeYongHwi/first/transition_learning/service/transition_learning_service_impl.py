import torch

from transition_learning.repository.transition_learning_repository_impl import TransitionLearningRepositoryImpl
from transition_learning.service.transition_learning_service import TransitionLearningService


class TransitionLearningServiceImpl(TransitionLearningService):
    def __init__(self):
        self.__transitionLearningRepository = TransitionLearningRepositoryImpl()

    def predictText(self, transitionLearningPredictRequestFormuestForm):
        # 전통적인 방식은 이전에 저희가 작업 했듯이 단어(word), 문자(character) 단위로 전부 학습을 시키고
        # 그 학습 정보를 바탕으로 다른 문장을 추론하거나 기타 작업들이 필요했음
        # 반면 Transformer를 사용할 경우 보다 편하고 수월하게 작업이 가능
        # BERT(Bidrectional Encodere Representation from Transformers)는
        # Trasformer 아키텍처를 기반으로 Pre Trained Model을 사용하는 기법
        # BERT 방식을 사용하면 문장의 문맥을 이해하는 능력이 뛰어나 의미 파악에 효율적
        modelName, tokenizer, model = self.__transitionLearningRepository.prepareBertBaseUncasedLearningSet()
        model.eval()

        # 입력한 두 문장을 위의 tokenizer를 통해 토큰화하여 모델에 입력
        # BERT는 문장을 토큰으로 변환하고, 각 토큰을 임베딩 벡터로 변환
        tokenizedUserInputText = tokenizer(
            transitionLearningPredictRequestFormuestForm.firstSentence,
            transitionLearningPredictRequestFormuestForm.secondSentence,
            return_tensors="pt",
            truncation=True
        )

        # 모델은 두 문장을 인코딩 한 이후 두 문장의 관계를 나타내는 벡터를 구성함
        # 이 벡터를 사용하여 두 문장이 의미적으로 동일한지 여부를 판단
        # 이전에는 이것을 직접 구현했었는데
        # Transformer 같은 것을 사용하면 이 부분은 자동으로 처리됨
        with torch.no_grad():
            outputList = model(**tokenizedUserInputText)
            logitList = outputList.logits
            prediction = torch.argmax(logitList, dim=1).item()

        if prediction == 1:
            result = "두 문장은 다르지만 의미는 같습니다."
        else:
            result = "두 문장의 의미는 다릅니다."

        return result

    def transitionLearningWithSentimentAnalysis(self, sentimentAnalysisRequestForm):
        modelName, tokenizer, model = (
            self.__transitionLearningRepository.prepareBertBaseMultilingualUncasedSentimentLearningSet())
        model.eval()

        tokenizedUserInputText = tokenizer(
            sentimentAnalysisRequestForm.sentence,
            return_tensors="pt",
            truncation=True
        )

        with torch.no_grad():
            outputList = model(**tokenizedUserInputText)
            logitList = outputList.logits
            prediction = torch.argmax(logitList, dim=1).item()

            sentimentList = ["매우 부정", "부정", "중립", "긍정", "매우 긍정"]
            sentiment = sentimentList[prediction]

            return sentiment

    def predictTextWithGPT2(self, gpt2PretrainedPredictionRequestForm):
        modelName, tokenizer, model = (
            self.__transitionLearningRepository.prepareGPT2PretrainedLearningSet())
        model.eval()

        # tokenizedUserInputText = tokenizer(
        #     gpt2PretrainedPredictionRequestForm.text,
        #     return_tensors="pt",
        # )

        encodedInputList = tokenizer.encode(gpt2PretrainedPredictionRequestForm.text, return_tensors="pt")

        outputList = model.generate(encodedInputList, max_length=200, num_return_sequences=1)
        generatedText = tokenizer.decode(outputList[0], skip_special_tokens=True)

        return generatedText


from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from transition_learning.controller.request_form.gpt2_pretrained_prediction_request_form import \
    GPT2PretrainedPredictionRequestForm
from transition_learning.controller.request_form.sentiment_analysis_request_form import SentimentAnalysisRequestForm
from transition_learning.controller.request_form.transition_learning_predict_request_form import \
    TransitionLearningPredictRequestForm
from transition_learning.service.transition_learning_service_impl import TransitionLearningServiceImpl

transitionLearningRouter = APIRouter()

async def injectTransitionLearningService() -> TransitionLearningServiceImpl:
    return TransitionLearningServiceImpl()


@transitionLearningRouter.post("/transition-learning-predict")
async def predictWithTransitionLearning(transitionLearningPredictReqeustForm: TransitionLearningPredictRequestForm,
                                        transitionLearningService: TransitionLearningServiceImpl =
                                        Depends(injectTransitionLearningService)):

    print(f"controller -> predictWithTransitionLearning(): "
          f"transitionLearningPredictRequestForm: {transitionLearningPredictReqeustForm}")

    predictedResult = transitionLearningService.predictText(transitionLearningPredictReqeustForm)

    return JSONResponse(content=predictedResult, status_code=status.HTTP_200_OK)

@transitionLearningRouter.post("/transition-learning-with-sentiment-analysis")
async def transitionLearningWithSentimentAnalysis(sentimentAnalysisRequestForm: SentimentAnalysisRequestForm,
                                                  transitionLearningService: TransitionLearningServiceImpl =
                                                  Depends(injectTransitionLearningService)):
    print(f"controller -> predictWithTransitionLearning(): "
          f"sentimentAnalysisRequestForm: {sentimentAnalysisRequestForm}")

    predictedResult = transitionLearningService.transitionLearningWithSentimentAnalysis(sentimentAnalysisRequestForm)

    return JSONResponse(content=predictedResult, status_code=status.HTTP_200_OK)

@transitionLearningRouter.post("/gpt2-pretrained-prediction")
async def gpt2PretrainedPrediction(gpt2PretrainedPredictionRequestForm: GPT2PretrainedPredictionRequestForm,
                                   transitionLearningService: TransitionLearningServiceImpl =
                                   Depends(injectTransitionLearningService)):
    print(f"controller -> gpt2PretrainedPrediction(): "
          f"gpt2PretrainedPredictionRequestForm: {gpt2PretrainedPredictionRequestForm}")

    predictedResult = transitionLearningService.predictTextWithGPT2(gpt2PretrainedPredictionRequestForm)

    return JSONResponse(content=predictedResult, status_code=status.HTTP_200_OK)
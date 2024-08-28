import os

import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import HTTPException
import openai
import faiss
from langchain.chains.llm import LLMChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from motor.motor_asyncio import AsyncIOMotorDatabase

from openai_basic.repository.openai_basic_repository import OpenAIBasicRepository


load_dotenv()

openaiApiKey = os.getenv('OPENAI_API_KEY')
if not openaiApiKey:
    raise ValueError('API Key가 준비되어 있지 않습니다.')

class OpenAIBasicRepositoryImpl(OpenAIBasicRepository):
    SIMILARITY_TOP_RANK = 3

    headers = {
        'Authorization': f'Bearer {openaiApiKey}',
        'Content-Type': 'application/json'
    }

    templateQuery = """You are a helpful assistant.
    {question}
    Provide a detailed answer to the above question."""

    OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, vectorDbPool: AsyncIOMotorDatabase):
        self.vectorDbPool = vectorDbPool
        self.embeddingModel = OpenAIEmbeddings(model="text-embedding-ada-002")

    async def generateText(self, userSendMessage):
        data = {
            'model': 'ft:gpt-4o-mini-2024-07-18:personal::9ywxDb8p',
            'messages': [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": userSendMessage}
            ]
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.OPENAI_CHAT_COMPLETIONS_URL, headers=self.headers, json=data)
                response.raise_for_status()

                return response.json()['choices'][0]['message']['content'].strip()

            except httpx.HTTPStatusError as e:
                print(f"HTTP Error: {str(e)}")
                print(f"Status Code: {e.response.status_code}")
                print(f"Response Text: {e.response.text}")
                raise HTTPException(status_code=e.response.status_code, detail=f'HTTP Error: {e}')

            except (httpx.RequestError, ValueError) as e:
                print(f"Request Error: {e}")
                raise HTTPException(status_code=500, detail=f"Request Error: {e}")

    async def sentimentAnalysis(self, userSendMessage):
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. 한글로 답변하자!" },
                {"role": "user", "content": f"Analyze the sentiment of the following text:\n\n{userSendMessage}"}
            ]
        )
        print(f"openai response: {response.json()}")
        return response.choices[0].message.content.strip()

    def audioAnalysis(self, audioFile):
        try:
            # 임시 파일 저장
            fileLocation = f"temp_{audioFile.filename}"
            with open(fileLocation, "wb+") as fileObject:
                fileObject.write(audioFile.file.read())

            # Whisper API 호출
            with open(fileLocation, "rb") as file:
                transcript = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=file
                )

            # 임시 파일 삭제
            os.remove(fileLocation)

            return transcript.text

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Am error occurred: {str(e)}")

    def embeddingList(self):
        return self.vectorDbPool['embeddings']

    def openAiBasedEmbedding(self, paperTitleList):
        response = openai.embeddings.create(
            input=paperTitleList,
            model="text-embedding-ada-002"
        )

        print(f"response: {response}")
        return response.data[0].embedding

    def createL2FaissIndex(self, embeddingVectorDimension):
        return faiss.IndexFlatL2(embeddingVectorDimension)

    def similarityAnalysis(self, userRequestPaperTitle, faissIndex):
        embeddingUserRequest = np.array(
            self.openAiBasedEmbedding(userRequestPaperTitle)).astype('float32').reshape(1, -1)
        distanceList, indexList = faissIndex.search(embeddingUserRequest, self.SIMILARITY_TOP_RANK)

        return indexList[0], distanceList[0]

    def faissIndexFromVector(self, embeddingList):
        return FAISS.from_embeddings(vectors=embeddingList, embedding=self.embeddingModel)

    def createPromptTemplate(self):
        return PromptTemplate(
            input_variable=["question"],
            template=self.templateQuery
        )

    def loadOpenAILLM(self):
        return ChatOpenAI(model="gpt-3.5-turbo")

    def createLLMChain(self, llm, prompt):
        return LLMChain(llm=llm, prompt=prompt)

    def runLLMChain(self, llmChain, userSendMessage):
        return llmChain.run(userSendMessage)





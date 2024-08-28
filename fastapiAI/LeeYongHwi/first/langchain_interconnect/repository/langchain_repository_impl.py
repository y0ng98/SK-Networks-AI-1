from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS

from langchain_interconnect.repository.langchain_repository import LangchainRepository
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain


class LangchainRepositoryImpl(LangchainRepository):
    DOCUMENT_FILE = 'documents.txt'

    def __init__(self):
        self.chatHistory = []
    def loadDocumentation(self):
        with open(self.DOCUMENT_FILE, "r") as file:
            text = file.read()

        return text

    def createTextChunk(self, documnetList, chunk_size=1000, chunk_overlap=0):
        chunks = []
        start = 0
        while start < len(documnetList):
            end = min(start + chunk_size, len(documnetList))
            chunks.append(documnetList[start:end])
            start += chunk_size - chunk_overlap
        return chunks

    def createFaissIndex(self, documentList):
        embeddings = OpenAIEmbeddings()
        return FAISS.from_texts(documentList, embeddings)

    def loadLLMChain(self):
        return ChatOpenAI(model_name="gpt-4")

    def createRegChain(self, llm, faissIndex):
        return ConversationalRetrievalChain.from_llm(llm, retriever=faissIndex.as_retriever())

    def runChain(self, chain, userSendMessage):
        response = chain.run({"question": userSendMessage, "chat_history": self.chatHistory})
        self.chatHistory.append({"question": userSendMessage, "answer": response})

        return response
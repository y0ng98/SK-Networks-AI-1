from fastapi import APIRouter, Depends, status, UploadFile, File

from fastapi.responses import JSONResponse

from langchain_interconnect.controller.request_form.langchain_request_form import LangchainRequestForm
from langchain_interconnect.service.langchain_service_impl import LangchainServiceImpl

langchainRouter = APIRouter()

async def injectLangchainService():
    return LangchainServiceImpl()

@langchainRouter.post("/reg-based-question")
async def regWithLangChain(langchainRequestForm: LangchainRequestForm,
                           langchainService: LangchainServiceImpl =
                           Depends(injectLangchainService)):

    print(f"controller -> regWithLangChain(): langchainRequestForm: {langchainRequestForm}")

    result = await langchainService.regWithLangchain(langchainRequestForm.userSendMessage)

    return JSONResponse(content={"result": result},status_code=status.HTTP_200_OK)
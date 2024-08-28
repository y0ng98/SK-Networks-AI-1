from fastapi import APIRouter, status, Depends

from fastapi.responses import JSONResponse

from gdft.controller.request_form.gdft_user_request_form import GDFTUserRequestForm
from gdft.service.gdft_service_impl import GDFTServiceImpl

gameDataFineTuningRouter = APIRouter()

async def injectGDFTService() -> GDFTServiceImpl:
    return GDFTServiceImpl()

@gameDataFineTuningRouter.post("/gdft")
async def gdftTest(gdftService: GDFTServiceImpl = Depends(injectGDFTService)):
    print(f"controller -> gdftTest()")

    gdftService.gdftTest()

@gameDataFineTuningRouter.post("/gdft-user-request")
async def gdftUserRequestTest(userRequestForm: GDFTUserRequestForm,
                              gdftService: GDFTServiceImpl = Depends(injectGDFTService)):
    print(f"controller -> gdftUserRequestTest()")

    response = gdftService.gdftUserRequestTest(userRequestForm.text)

    return JSONResponse(content=response, status_code=status.HTTP_200_OK)
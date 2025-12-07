from fastapi import APIRouter
from models.models import AliveResponse

router = APIRouter()

@router.get("/alive", response_model=AliveResponse)
async def health_check():
    return AliveResponse(status="alive")
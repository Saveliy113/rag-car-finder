from pydantic import BaseModel

class RagQueryRequest(BaseModel):
    question: str
    top_k: int = 5

class AliveResponse(BaseModel):
    status: str
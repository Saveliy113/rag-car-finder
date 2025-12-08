from pydantic import BaseModel, Field, field_validator

class RagQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The search query/question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return (1-20)")

    @field_validator('question')
    @classmethod
    def validate_question(cls, v):
        if not v or not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()

class RagQueryResponse(BaseModel):
    data: str

class AliveResponse(BaseModel):
    status: str
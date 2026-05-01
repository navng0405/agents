from typing import Any

from pydantic import BaseModel, EmailStr, Field


class UserProfile(BaseModel):
    user_id: str
    email: EmailStr
    full_name: str | None = None
    preferences: dict[str, Any] = Field(default_factory=dict)


class BrainRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question or note to process")


class BrainResponse(BaseModel):
    answer: str
    intent: str
    retrieved_notes: list[str] = Field(default_factory=list)
    workflow_steps: list[str] = Field(default_factory=list)

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class DocumentBase(BaseModel):
    TweetID: float
    CreateDate: datetime
    Antisemitic: bool
    text: str
    emotion: Optional[str] = None
    weapons: Optional[List[str]] = None


class DocumentCreate(DocumentBase):
    pass


class DocumentUpdate(BaseModel):
    TweetID: Optional[float] = None
    CreateDate: Optional[datetime] = None
    Antisemitic: Optional[bool] = None
    text: Optional[str] = None
    emotion: Optional[str] = None
    weapons: Optional[List[str]] = None


class DocumentResponse(DocumentBase):
    id: str
    created_at: datetime
    updated_at: datetime
    emotion: Optional[str] = None
    weapons: Optional[List[str]] = None

    class Config:
        from_attributes = True


class SearchResponse(BaseModel):
    total_hits: int
    max_score: Optional[float]
    took_ms: int
    documents: List[DocumentResponse]


class BulkOperationResponse(BaseModel):
    success_count: int
    error_count: int
    errors: List[str]

# Backend/schemas.py
from __future__ import annotations
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

# 이미 존재한다면 유지
class PredictResponse(BaseModel):
    id: int
    class_name: str
    confidence: float
    recomm: str
    image_path: str
    sources: List[Dict[str, Any]] = []
    detailed_prediction: Dict[str, Any] | None = None

# 신규: 리스트 아이템(간략)
class ResultItem(BaseModel):
    id: int
    class_name: str
    image_path: str
    created_at: str  # ISO 문자열로 반환

# 신규: 페이지네이션 응답
class ResultsPage(BaseModel):
    total: int
    page: int = Field(ge=1)
    size: int = Field(ge=1, le=200)
    items: List[ResultItem]

# 신규: 상세조회
class ResultDetail(BaseModel):
    id: int
    class_name: str
    recomm: str
    image_path: str
    created_at: str
    updated_at: str
    # class_info는 JSON 문자열이므로, 파싱해서 dict로 반환
    class_info: Dict[str, Any] | None = None

# 신규: 삭제 응답
class DeleteResult(BaseModel):
    id: int
    deleted: bool

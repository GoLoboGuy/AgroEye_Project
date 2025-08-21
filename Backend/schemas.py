# Backend/schemas.py
from __future__ import annotations
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

# 소스 정보 아이템 (문자열로 주던 걸 dict로 바꿨을 때 매핑)
class SourceItem(BaseModel):
    source: str
    page: Optional[int] = None
    score: Optional[float] = None
    snippet: Optional[str] = None
    title: Optional[str] = None

class PredictResponse(BaseModel):
    id: int
    class_name: str
    confidence: float
    recomm: str
    image_path: str
    sources: List[SourceItem] = Field(default_factory=list)  # ✅ dict→모델
    detailed_prediction: Optional[Dict[str, Any]] = None

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
    class_info: Optional[Dict[str, Any]] = None

# 신규: 삭제 응답
class DeleteResult(BaseModel):
    id: int
    deleted: bool

# Backend/api.py
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import List, Any, Dict, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Path as FPath

from .config import settings
from .database import SessionLocal
from .models import FinalProjectResult
from .schemas import (
    PredictResponse,
    ResultsPage,
    ResultItem,
    ResultDetail,
    DeleteResult,
)
from .services.classifier import classifier
from .services.synonyms import class_to_query_terms, as_boolean_query
from .services.rag_service import rag, Retrieved

from .schemas import PredictResponse, SourceItem  # ← SourceItem 추가

# router = APIRouter(tags=["predict"])  # ← FastAPI 대신 APIRouter
router = APIRouter() # 기본 tags 제거

# RAG 결과 → SourceItem dict 변환 헬퍼 추가
def _to_source_item(hit) -> dict:
    """
    hit: Retrieved 객체 (rag.search()에서 반환)
    반환: SourceItem에 맞는 dict
    """
    # Retrieved 객체의 구조: text, meta, score
    meta = getattr(hit, "meta", {}) or {}
    score = getattr(hit, "score", None)
    
    src = str(meta.get("source", "")).replace("\\", "/")  # 경로 통일
    page = meta.get("page", None)
    try:
        page = int(page) if page is not None else None
    except Exception:
        page = None

    snippet = (getattr(hit, "text", "") or "")[:300]
    title = meta.get("title", None)

    return {
        "source": src,
        "page": page,
        "score": score,
        "snippet": snippet,
        "title": title,
    }

# -----------------------------
# 1) 예측
# -----------------------------
@router.post("/predict", response_model=PredictResponse, tags=["predict"])
async def predict(file: UploadFile = File(...)):
    # 1) 이미지 저장
    save_name = f"img_{os.getpid()}_{file.filename}"
    save_path = Path(settings.UPLOAD_DIR) / save_name
    data = await file.read()
    save_path.write_bytes(data)

    # 2) 분류 모델 추론 (상세 결과 포함)
    try:
        class_name, confidence = classifier.classify(str(save_path))
        detailed_result = classifier.classify_with_details(str(save_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"classifier error: {e}")

    # 3) Unknown 클래스인 경우 특별 처리
    if class_name == "Unknown":
        warning_message = (
            "⚠️ 신뢰도 부족으로 인한 분류 실패\n\n"
            "분류 모델의 신뢰도가 낮거나 모델 간 결과 차이가 커서 정확한 분류를 수행할 수 없습니다.\n\n"
            "권장사항:\n"
            "- 더 선명하고 품질이 좋은 이미지를 사용해주세요\n"
            "- 잎사귀가 이미지 중앙에 잘 보이도록 촬영해주세요\n"
            "- 다른 각도에서 촬영해보세요"
        )
        
        return PredictResponse(
            id=0,  # DB에 저장하지 않으므로 0
            class_name="Unknown",
            confidence=0.0,
            recomm=warning_message,
            image_path=str(save_path),
            sources=[],  # 소스 정보 없음
            detailed_prediction=detailed_result,
        )

    # 4) 클래스 → 동의어/한글 질의 변환
    terms = class_to_query_terms(class_name)
    boolean_query = as_boolean_query(terms)

    # 5) RAG 검색 + 설명 생성
    if not rag:
        raise HTTPException(status_code=500, detail="RAG 서비스가 초기화되지 않았습니다. 인덱스를 먼저 생성하세요.")
    retrieved: List[Retrieved] = rag.search(boolean_query, k=4)
    explanation = rag.generate_explanation(boolean_query, retrieved)

    # dict 리스트 (SourceItem 스키마에 맞춤)
    sources_dicts = [_to_source_item(h) for h in retrieved[:4]]
    # Pydantic 모델로 감싸도 되고(dict 그대로도 OK)
    sources_items = [SourceItem(**d) for d in sources_dicts]

    # 6) DB 저장 (JSON 직렬화에는 dict가 적합)
    class_info_obj = {
        "query_terms": terms,
        "boolean_query": boolean_query,
        "sources": sources_dicts,                   # ← dict 사용 (모델 말고)
        "detailed_prediction": detailed_result,
    }
    class_info = json.dumps(class_info_obj, ensure_ascii=False)

    db = SessionLocal()
    try:
        row = FinalProjectResult(
            class_name=class_name,
            class_info=class_info,
            recomm=explanation,
            image_path=str(save_path),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        row_id = row.id
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB 저장 실패: {e}")
    finally:
        db.close()

    return PredictResponse(
        id=row_id,
        class_name=class_name,
        confidence=confidence,
        recomm=explanation,
        image_path=str(save_path),
        sources=sources_items,
        detailed_prediction=detailed_result,
    )

# -----------------------------
# 2) 모델 상태
# -----------------------------
@router.get("/model/status", tags=["predict"])
async def get_model_status():
    """분류 모델의 상태 확인"""
    return {
        "model_loaded": getattr(classifier, "loaded", False),
        "model_available": getattr(classifier, "model_available", False),
        "status": "ready" if getattr(classifier, "loaded", False) else "not_loaded",
    }

# -----------------------------
# 3) 조회 (리스트)
#    GET /results?page=1&size=20&class_name=...&order=desc
# -----------------------------
@router.get("/results", response_model=ResultsPage, tags=["results"])
def list_results(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=200),
    class_name: Optional[str] = Query(None),
    order: str = Query("desc", pattern="^(asc|desc)$"),
):
    """
    결과 리스트 조회 (페이지네이션 + 선택적 class_name 필터)
    created_at 내림차순(desc) 기본
    """
    offset = (page - 1) * size
    db = SessionLocal()
    try:
        q = db.query(FinalProjectResult)
        if class_name:
            q = q.filter(FinalProjectResult.class_name == class_name)

        total = q.count()

        # 정렬
        if order.lower() == "asc":
            q = q.order_by(FinalProjectResult.created_at.asc())
        else:
            q = q.order_by(FinalProjectResult.created_at.desc())

        rows = q.offset(offset).limit(size).all()

        items: List[ResultItem] = []
        for r in rows:
            items.append(
                ResultItem(
                    id=r.id,
                    class_name=r.class_name,
                    image_path=r.image_path,
                    created_at=r.created_at.isoformat() if r.created_at else "",
                )
            )

        return ResultsPage(total=total, page=page, size=size, items=items)
    finally:
        db.close()

# -----------------------------
# 4) 상세조회
#    GET /results/{id}
# -----------------------------
@router.get("/results/{id}", response_model=ResultDetail, tags=["results"])
def get_result(
    id: int = FPath(..., ge=1),
):
    db = SessionLocal()
    try:
        r = db.query(FinalProjectResult).get(id)
        if not r:
            raise HTTPException(status_code=404, detail="Not Found")

        # class_info JSON 파싱
        try:
            info = json.loads(r.class_info) if r.class_info else None
        except json.JSONDecodeError:
            info = None

        return ResultDetail(
            id=r.id,
            class_name=r.class_name,
            recomm=r.recomm or "",
            image_path=r.image_path,
            created_at=r.created_at.isoformat() if r.created_at else "",
            updated_at=r.updated_at.isoformat() if r.updated_at else "",
            class_info=info,
        )
    finally:
        db.close()

# -----------------------------
# 5) 삭제
#    DELETE /results/{id}
# -----------------------------
@router.delete("/results/{id}", response_model=DeleteResult, tags=["results"])
def delete_result(
    id: int = FPath(..., ge=1),
):
    db = SessionLocal()
    try:
        r = db.query(FinalProjectResult).get(id)
        if not r:
            # 이미 없는 경우도 일단 false로 응답
            return DeleteResult(id=id, deleted=False)
        db.delete(r)
        db.commit()
        return DeleteResult(id=id, deleted=True)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"삭제 실패: {e}")
    finally:
        db.close()

# Backend/api.py
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import List, Any, Dict, Optional
from datetime import datetime

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

from .schemas import PredictResponse, SourceItem

router = APIRouter()

def generate_unique_filename(original_filename: str, upload_dir: Path) -> str:
    """
    파일명을 "파일이름_YYMMDD_000.jpg" 형태로 생성하고 숫자를 000~999까지 순차적으로 증가시킴
    """
    # 파일 확장자 분리
    name, ext = os.path.splitext(original_filename)
    if not ext:
        ext = ".jpg"  # 기본 확장자
    
    # 현재 날짜 (YYMMDD 형식)
    current_date = datetime.now().strftime("%y%m%d")
    
    # 기본 파일명 생성
    base_filename = f"{name}_{current_date}"
    
    # 000부터 999까지 순차적으로 확인하여 사용 가능한 번호 찾기
    for i in range(1000):
        number_str = f"{i:03d}"  # 000, 001, 002, ..., 999
        filename = f"{base_filename}_{number_str}{ext}"
        file_path = upload_dir / filename
        
        # 파일이 존재하지 않으면 해당 파일명 사용
        if not file_path.exists():
            return filename
    
    # 999까지 모두 사용된 경우 (극히 드문 경우)
    raise HTTPException(status_code=500, detail="파일 저장 공간이 가득 찼습니다. (999개 파일 초과)")

def _to_source_item(hit) -> dict:
    meta = getattr(hit, "meta", {}) or {}
    score = getattr(hit, "score", None)
    src = str(meta.get("source", "")).replace("\\", "/")
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

@router.post("/predict", response_model=PredictResponse, tags=["predict"])
async def predict(file: UploadFile = File(...)):
    # 1) 이미지 저장
    save_name = generate_unique_filename(file.filename, Path(settings.UPLOAD_DIR))
    save_path = Path(settings.UPLOAD_DIR) / save_name
    data = await file.read()
    save_path.write_bytes(data)

    # 2) 분류
    try:
        class_name, confidence = classifier.classify(str(save_path))
        detailed_result = classifier.classify_with_details(str(save_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"classifier error: {e}")

    # 3) Unknown 처리 (RAG 생략)
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
            id=0,
            class_name="Unknown",
            confidence=0.0,
            recomm=warning_message,
            image_path=str(save_path),
            sources=[],
            detailed_prediction=detailed_result,
        )

    # 4) 클래스 질의어 생성
    terms = class_to_query_terms(class_name)
    boolean_query = as_boolean_query(terms)

    # 5) RAG (인덱스 필수)
    if not rag:
        raise HTTPException(status_code=500, detail="RAG 서비스가 초기화되지 않았습니다. 인덱스를 먼저 생성하세요.")
    retrieved: List[Retrieved] = rag.search(boolean_query, k=4)
    explanation = rag.generate_explanation(boolean_query, retrieved)

    sources_dicts = [_to_source_item(h) for h in retrieved[:4]]
    sources_items = [SourceItem(**d) for d in sources_dicts]

    # 6) DB 저장
    class_info_obj = {
        "query_terms": terms,
        "boolean_query": boolean_query,
        "sources": sources_dicts,
        "detailed_prediction": detailed_result,   # ← 디버깅에 유용
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

@router.get("/model/status", tags=["predict"])
async def get_model_status():
    return {
        "model_loaded": getattr(classifier, "loaded", False),
        "model_available": getattr(classifier, "model_available", False),
        "status": "ready" if getattr(classifier, "loaded", False) else "not_loaded",
    }

@router.get("/results", response_model=ResultsPage, tags=["results"])
def list_results(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=200),
    class_name: Optional[str] = Query(None),
    order: str = Query("desc", pattern="^(asc|desc)$"),
):
    offset = (page - 1) * size
    db = SessionLocal()
    try:
        q = db.query(FinalProjectResult)
        if class_name:
            q = q.filter(FinalProjectResult.class_name == class_name)

        total = q.count()

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

@router.get("/results/{id}", response_model=ResultDetail, tags=["results"])
def get_result(id: int = FPath(..., ge=1)):
    db = SessionLocal()
    try:
        r = db.query(FinalProjectResult).get(id)
        if not r:
            raise HTTPException(status_code=404, detail="Not Found")
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

@router.delete("/results/{id}", response_model=DeleteResult, tags=["results"])
def delete_result(id: int = FPath(..., ge=1)):
    db = SessionLocal()
    try:
        r = db.query(FinalProjectResult).get(id)
        if not r:
            return DeleteResult(id=id, deleted=False)
        db.delete(r)
        db.commit()
        return DeleteResult(id=id, deleted=True)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"삭제 실패: {e}")
    finally:
        db.close()

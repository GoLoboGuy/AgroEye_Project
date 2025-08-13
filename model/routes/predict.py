# model/routes/predict.py
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from typing import List
from PIL import Image
import io
import os
import sys
from datetime import datetime
from pathlib import Path

# Backend 모듈 경로 추가
backend_path = Path(__file__).parent.parent.parent / "Backend"
sys.path.append(str(backend_path))

# leaf_ensemble을 절대 경로로 import
from leaf_ensemble import get_model
from database import get_db
from crud import save_result
from sqlalchemy.orm import Session

router = APIRouter(prefix="/api", tags=["inference"])

@router.post("/predict")
async def predict(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """
    이미지를 업로드하여 잎사귀 질병을 예측하고 결과를 데이터베이스에 저장합니다.
    """
    model = get_model()
    results = []
    
    for f in files:
        try:
            # 파일 읽기
            raw = await f.read()
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
            
            # 모델 예측
            prediction = model.predict_one(pil)
            
            # 이미지 저장 경로 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{f.filename}"
            # Backend 폴더 내의 images 폴더에 저장
            backend_images_dir = Path(__file__).parent.parent.parent / "Backend" / "images"
            image_path = str(backend_images_dir / filename)
            
            # 이미지 저장
            os.makedirs(backend_images_dir, exist_ok=True)
            pil.save(image_path)
            
            # 예측 결과에서 선택된 클래스명 추출
            selected_class = prediction["picked"]["label"]
            
            # 클래스 정보 및 추천사항 생성 (실제로는 더 정교한 로직이 필요할 수 있음)
            class_info = f"예측 모델: {prediction['picked']['model']}, 신뢰도: {prediction['picked']['confidence']:.3f}"
            
            # 간단한 추천사항 (실제로는 더 정교한 로직이 필요할 수 있음)
            if "healthy" in selected_class.lower():
                recomm = "건강한 상태입니다. 정기적인 관수를 유지하세요."
            elif "disease" in selected_class.lower():
                recomm = "질병이 의심됩니다. 전문가 상담을 권장합니다."
            else:
                recomm = "정기적인 모니터링이 필요합니다."
            
            # 데이터베이스에 결과 저장
            db_result = save_result(
                db=db,
                class_name=selected_class,
                class_info=class_info,
                recomm=recomm,
                image_path=image_path
            )
            
            # 응답용 결과 구성
            result = {
                "filename": f.filename,
                "prediction": prediction,
                "saved_result": {
                    "id": db_result.id,
                    "class_name": db_result.class_name,
                    "class_info": db_result.class_info,
                    "recomm": db_result.recomm,
                    "image_path": db_result.image_path,
                    "created_at": db_result.created_at
                }
            }
            results.append(result)
            
        except Exception as e:
            # 에러 발생 시 해당 파일만 실패 처리
            results.append({
                "filename": f.filename,
                "error": str(e),
                "success": False
            })
    
    return {"results": results}

@router.get("/results")
async def get_results(
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    최근 예측 결과들을 조회합니다.
    """
    # crud 모듈에서 함수 직접 import
    from crud import get_recent_results
    
    try:
        results = get_recent_results(db, limit=limit)
        return {
            "results": [
                {
                    "id": result.id,
                    "class_name": result.class_name,
                    "class_info": result.class_info,
                    "recomm": result.recomm,
                    "image_path": result.image_path,
                    "created_at": result.created_at,
                    "updated_at": result.updated_at
                }
                for result in results
            ],
            "total_count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"결과 조회 실패: {str(e)}")

@router.get("/results/{result_id}")
async def get_result_by_id(
    result_id: int,
    db: Session = Depends(get_db)
):
    """
    특정 ID의 예측 결과를 조회합니다.
    """
    # crud 모듈에서 함수 직접 import
    from crud import get_result_by_id
    
    try:
        result = get_result_by_id(db, result_id)
        if not result:
            raise HTTPException(status_code=404, detail="결과를 찾을 수 없습니다")
        
        return {
            "id": result.id,
            "class_name": result.class_name,
            "class_info": result.class_info,
            "recomm": result.recomm,
            "image_path": result.image_path,
            "created_at": result.created_at,
            "updated_at": result.updated_at
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"결과 조회 실패: {str(e)}")

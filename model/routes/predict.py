# model/routes/predict.py
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from typing import List
from PIL import Image
import io
import os
from datetime import datetime
from pathlib import Path

# leaf_ensemble을 상대 경로로 import
from ..leaf_ensemble import get_model

router = APIRouter(prefix="/api", tags=["inference"])

@router.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    """
    이미지를 업로드하여 잎사귀 질병을 예측합니다.
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
            
            # 현재 프로젝트의 images 폴더에 저장
            current_dir = Path(__file__).parent.parent
            images_dir = current_dir / "images"
            os.makedirs(images_dir, exist_ok=True)
            image_path = str(images_dir / filename)
            
            # 이미지 저장
            pil.save(image_path)
            
            # 응답용 결과 구성
            result = {
                "filename": f.filename,
                "prediction": prediction,
                "image_path": image_path,
                "success": True
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

@router.get("/model/status")
async def get_model_status():
    """
    모델의 상태를 확인합니다.
    """
    try:
        model = get_model()
        return {
            "status": "ready",
            "model_type": "LeafEnsemble",
            "models": ["MobileNetV2", "ResNet50"],
            "device": str(model.mn.device)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

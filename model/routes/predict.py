# backend/routes/predict.py
from fastapi import APIRouter, UploadFile, File
from typing import List
from PIL import Image
import io

from ..leaf_ensemble import get_model

router = APIRouter(prefix="/api", tags=["inference"])

@router.get("/health")
def health():
    return {"ok": True}

@router.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    model = get_model()
    results = []
    for f in files:
        raw = await f.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        out = model.predict_one(pil)
        out["filename"] = f.filename
        results.append(out)
    return {"results": results}

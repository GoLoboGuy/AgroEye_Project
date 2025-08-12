from pydantic import BaseModel
from datetime import datetime
from typing import List

class ImageAnalysisBase(BaseModel):
    id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    image_path: str
    created_at: datetime

    class Config:
        orm_mode = True

class ImageAnalysisList(BaseModel):
    results: List[ImageAnalysisBase]

from pydantic import BaseModel
from typing import List
from datetime import datetime

class Detection(BaseModel):
    bbox: List[float]
    confidence: float
    label: str

class AnalysisCreate(BaseModel):
    detections: List[Detection]

class Analysis(AnalysisCreate):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True

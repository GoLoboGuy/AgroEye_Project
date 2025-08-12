from .database import SessionLocal
from . import models, schemas

def create_analysis(detections: list[schemas.Detection]):
    db = SessionLocal()
    obj = models.ImageAnalysis(results=[d.dict() for d in detections])
    db.add(obj)
    db.commit()
    db.refresh(obj)
    db.close()
    return obj

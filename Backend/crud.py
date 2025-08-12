from sqlalchemy.orm import Session
from models import ImageAnalysis

def save_result(db: Session, data: dict):
    item = ImageAnalysis(
        class_name=data["class"],
        confidence=data["confidence"],
        x1=data["x1"],
        y1=data["y1"],
        x2=data["x2"],
        y2=data["y2"],
        image_path=data.get("image_path", "")
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item

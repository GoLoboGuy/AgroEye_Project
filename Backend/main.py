from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from database import SessionLocal
from models import ImageAnalysis
from schemas import ImageAnalysisBase, ImageAnalysisList
from crud import save_result
from yolo_model import run_inference

app = FastAPI(title="YOLO Image Analysis API", version="1.0.0")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/results", response_model=ImageAnalysisList)
def get_results(limit: int = 10, db: Session = Depends(get_db)):
    records = db.query(ImageAnalysis).order_by(ImageAnalysis.id.desc()).limit(limit).all()
    return {"results": records}

@app.get("/result/{result_id}", response_model=ImageAnalysisBase)
def get_result(result_id: int, db: Session = Depends(get_db)):
    record = db.query(ImageAnalysis).filter(ImageAnalysis.id == result_id).first()
    if not record:
        raise HTTPException(404, "Result not found")
    return record

@app.post("/upload")
async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    result = await run_inference(file)
    for item in result["results"]:
        save_result(db, item)
    return result

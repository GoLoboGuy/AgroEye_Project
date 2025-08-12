from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from . import image_utils, yolo_model, crud, schemas

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

@app.post("/api/detect", response_model=schemas.Analysis)
async def detect(file: UploadFile = File(...)):
    data = await file.read()
    img = image_utils.read_image(data)
    detections, annotated = yolo_model.infer(img)
    record = crud.create_analysis(detections)
    return record

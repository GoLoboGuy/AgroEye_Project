from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from database import Base

class ImageAnalysis(Base):
    __tablename__ = "image_analysis"
    id = Column(Integer, primary_key=True, index=True)
    class_name = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    x1 = Column(Float, nullable=False)
    y1 = Column(Float, nullable=False)
    x2 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)
    image_path = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

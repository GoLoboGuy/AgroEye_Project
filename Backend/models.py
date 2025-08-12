from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, JSON, DateTime, func

Base = declarative_base()

class ImageAnalysis(Base):
    __tablename__ = "image_analysis"
    id = Column(Integer, primary_key=True, index=True)
    results = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

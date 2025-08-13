# backend/models.py
from sqlalchemy.orm import declarative_base
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, func
)

Base = declarative_base()

class finalprojectresults(Base):
    __tablename__ = "final_project_results"

    id          = Column(Integer, primary_key=True, autoincrement=True, index=True)
    class_name  = Column(String(255), nullable=False, index=True)
    class_info  = Column(Text, nullable=True)
    recomm      = Column(Text, nullable=True)
    image_path  = Column(String(500), nullable=False)
    created_at  = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at  = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


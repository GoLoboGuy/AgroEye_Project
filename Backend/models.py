# Backend/models.py
from sqlalchemy import Column, Integer, String, Text, DateTime, Index, text
from .database import Base

class FinalProjectResult(Base):
    __tablename__ = "final_project_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    class_name = Column(String(255), nullable=False)
    class_info = Column(Text, nullable=True)
    recomm = Column(Text, nullable=True)
    image_path = Column(String(500), nullable=False)
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))
    updated_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"))

Index("idx_class_name", FinalProjectResult.class_name)
Index("idx_created_at", FinalProjectResult.created_at)
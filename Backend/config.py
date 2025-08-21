# Backend/config.py
from __future__ import annotations
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent  # Backend/ 디렉토리

class Settings(BaseSettings):
    DATABASE_URL: str
    OPENAI_API_KEY: str | None = None
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    RAG_INDEX_DIR: str = "Backend/rag/indexes/faiss"
    DOCS_DIR: str = "Backend/rag/docs"
    UPLOAD_DIR: str = "Backend/uploads"

    # pydantic-settings v2 방식
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),          # Backend/.env 를 명시
        env_file_encoding="utf-8",
        extra="ignore",                           # 여분 키는 무시
    )

settings = Settings()

# Backend/config.py
from __future__ import annotations
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator

BASE_DIR = Path(__file__).resolve().parent  # .../Backend

class Settings(BaseSettings):
    DATABASE_URL: str
    OPENAI_API_KEY: str | None = None
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # ✅ 기본값은 반드시 BASE_DIR 기준 상대경로(루트명 제거)
    RAG_INDEX_DIR: Path = Path("rag/indexes/faiss")
    DOCS_DIR: Path      = Path("rag/docs")
    UPLOAD_DIR: Path    = Path("uploads")

    @field_validator("RAG_INDEX_DIR", "DOCS_DIR", "UPLOAD_DIR", mode="before")
    @classmethod
    def make_abs(cls, v):
        p = Path(v) if not isinstance(v, Path) else v
        if p.is_absolute():
            return p
        # ✅ 'Backend/...'로 시작하면, BASE_DIR의 부모에 붙여 2중 결합 방지
        if p.parts and p.parts[0].lower() == BASE_DIR.name.lower():
            return (BASE_DIR.parent / p).resolve()
        # 일반 상대경로는 BASE_DIR 기준
        return (BASE_DIR / p).resolve()

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()

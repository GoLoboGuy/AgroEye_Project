# backend/config.py
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # 기본 설정
    APP_NAME: str = "Image Analysis API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False)
    
    # 서버 설정
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    
    # 데이터베이스 설정
    DATABASE_URL: str = Field(default="sqlite:///./test.db")
    
    # 이미지 처리 설정
    MAX_IMAGE_SIZE: int = Field(default=4096)
    JPEG_QUALITY: int = Field(default=85)
    MAX_FILE_SIZE: int = Field(default=10 * 1024 * 1024)  # 10MB
    SUPPORTED_FORMATS: List[str] = Field(
        default=[".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    )
    
    # YOLO 모델 설정
    MODEL_PATH: str = Field(default="yolov8n.pt")
    CONFIDENCE_THRESHOLD: float = Field(default=0.25)
    IOU_THRESHOLD: float = Field(default=0.45)
    MAX_DETECTIONS: int = Field(default=100)
    
    # 로깅 설정
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # CORS 설정
    CORS_ORIGINS: List[str] = Field(default=["*"])
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True)
    CORS_ALLOW_METHODS: List[str] = Field(default=["*"])
    CORS_ALLOW_HEADERS: List[str] = Field(default=["*"])
    
    # 데이터베이스 풀 설정
    DB_POOL_SIZE: int = Field(default=10)
    DB_MAX_OVERFLOW: int = Field(default=20)
    DB_POOL_RECYCLE: int = Field(default=3600)
    
    # 이미지 저장 설정
    IMAGES_DIR: str = Field(default="images")
    IMAGE_CLEANUP_DAYS: int = Field(default=30)
    
    # API 설정
    API_PREFIX: str = Field(default="/api/v1")
    DEFAULT_PAGE_SIZE: int = Field(default=100)
    MAX_PAGE_SIZE: int = Field(default=1000)
    
    # 환경 변수 파일(.env) 로드 설정 (Pydantic v2)
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

# 전역 설정 인스턴스
settings = Settings()

def get_settings() -> Settings:
    """설정 인스턴스 반환"""
    return settings 
# backend/main.py
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# model 모듈 경로 추가 (상위 디렉토리의 model 폴더)
project_root = Path(__file__).parent.parent
model_path = project_root / "model"
sys.path.append(str(model_path))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from database import init_db, test_db_connection
from schemas import HealthCheck

# 설정 로드
settings = get_settings()

# 로깅 설정
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# lifespan 이벤트 핸들러 정의 (on_event 대체)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    애플리케이션 시작 및 종료 이벤트 핸들러
    """
    logger.info("애플리케이션 시작 중...")

    # 테스트 환경에서는 데이터베이스 초기화 건너뛰기
    if not os.getenv("TESTING"):
        try:
            init_db()
            if not test_db_connection():
                logger.error("데이터베이스 연결 실패")
        except Exception as e:
            logger.error(f"데이터베이스 초기화 실패: {str(e)}")
    else:
        logger.info("테스트 환경 - 데이터베이스 초기화 건너뜀")

    yield  # 이 지점을 기준으로 위는 startup, 아래는 shutdown 코드

    logger.info("애플리케이션 종료 중...")
    logger.info("애플리케이션 종료 완료")


app = FastAPI(
    title=settings.APP_NAME,
    description="잎사귀 질병 감지 API",
    version=settings.APP_VERSION,
    lifespan=lifespan  # lifespan 이벤트 핸들러 등록
)

# Register classification routes
from routes.predict import router as predict_router
app.include_router(predict_router)

# CORS 설정 최적화
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

@app.get("/health", response_model=HealthCheck)
async def health_check_endpoint():
    """헬스 체크 엔드포인트"""
    try:
        # 데이터베이스 연결 확인
        db_status = test_db_connection()

        return {
            "status": "healthy" if db_status else "unhealthy",
            "service": "leaf-disease-detection-api",
            "timestamp": "2024-01-01T12:00:00Z",
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"헬스 체크 실패: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "leaf-disease-detection-api",
            "timestamp": "2024-01-01T12:00:00Z",
            "version": "1.0.0"
        }
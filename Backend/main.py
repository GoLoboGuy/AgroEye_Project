# Backend/main.py
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import structlog

from config import settings

APP_NAME = getattr(settings, "APP_NAME", "Leaf-Disease-Detection-API")
APP_VERSION = getattr(settings, "APP_VERSION", "1.0.0")
DEBUG = bool(getattr(settings, "DEBUG", True))
API_PREFIX = getattr(settings, "API_PREFIX", "/api")
if not API_PREFIX.startswith("/"):
    API_PREFIX = f"/{API_PREFIX}"

CORS_ORIGINS = getattr(settings, "CORS_ORIGINS", ["*"])
CORS_ALLOW_CREDENTIALS = bool(getattr(settings, "CORS_ALLOW_CREDENTIALS", True))
CORS_ALLOW_METHODS = getattr(settings, "CORS_ALLOW_METHODS", ["*"])
CORS_ALLOW_HEADERS = getattr(settings, "CORS_ALLOW_HEADERS", ["*"])

UPLOAD_DIR = getattr(settings, "UPLOAD_DIR", "Backend/uploads")
DOCS_DIR = getattr(settings, "DOCS_DIR", "Backend/rag/docs")
RAG_INDEX_DIR = getattr(settings, "RAG_INDEX_DIR", "Backend/rag/indexes/faiss")

# 선택: classifier 로드 지원
classifier = None
try:
    from services.classifier import Classifier
    classifier = Classifier()
except Exception as e:
    print(f"⚠️ classifier 준비 실패: {e} (스텁 모드)")

# DB 헬스체크
try:
    from database import test_db_connection
except Exception as e:
    print(f"⚠️ database import 경고: {e}")

    def test_db_connection() -> bool:
        return False

# 로깅
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("애플리케이션 시작 중...")
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        os.makedirs(DOCS_DIR, exist_ok=True)
        os.makedirs(RAG_INDEX_DIR, exist_ok=True)
        if classifier is not None:
            try:
                classifier.load()
                logger.info("분류 모델 로드 성공")
            except Exception as e:
                logger.warning("분류 모델 로드 실패 - 스텁 사용", error=str(e))
    except Exception as e:
        logger.error("초기화 실패", error=str(e))
        raise
    yield
    logger.info("애플리케이션 종료 중...")
    logger.info("애플리케이션 종료 완료")

app = FastAPI(
    title=APP_NAME,
    description=APP_NAME,
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if DEBUG else None,
    redoc_url="/redoc" if DEBUG else None,
)

# 미들웨어
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
)

# 요청 시간 측정
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.time() - start:.6f}"
    return response

# 🔗 API 라우터 연결 (안전 가드 포함)
try:
    from api import router as api_router
    app.include_router(api_router, prefix=API_PREFIX)
    logger.info("API 라우터 등록 성공", prefix=API_PREFIX)
except Exception as exc:
    logger.error("API 라우터 등록 실패", error=str(exc))
    fallback = APIRouter(tags=["api"])

    @fallback.get("/health")
    async def api_health_fallback():
        return {"status": "api_import_failed", "error": str(exc)}

    app.include_router(fallback, prefix=API_PREFIX)

# HealthCheck 모델 보완
try:
    from schemas import HealthCheck
except Exception:
    from pydantic import BaseModel

    class HealthCheck(BaseModel):
        status: str
        service: str
        timestamp: str
        version: str

@app.get("/health", response_model=HealthCheck)
async def health_check_endpoint():
    try:
        db_ok = test_db_connection()
        return {
            "status": "healthy" if db_ok else "unhealthy",
            "service": APP_NAME,
            "timestamp": "2024-01-01T12:00:00Z",
            "version": APP_VERSION,
        }
    except Exception as e:
        logger.error("헬스 체크 실패", error=str(e))
        return {
            "status": "unhealthy",
            "service": APP_NAME,
            "timestamp": "2024-01-01T12:00:00Z",
            "version": APP_VERSION,
        }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        "전역 예외 발생",
        error=str(exc),
        path=str(request.url.path),
        method=request.method,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "내부 서버 오류가 발생했습니다."}
    )

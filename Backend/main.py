# backend/main.py
import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

import crud
from config import get_settings
from database import get_db, init_db, test_db_connection
from image_utils import read_image, save_annotated_image
from schemas import (
    ImageAnalysisBase, ImageAnalysisList, AnalysisStatistics,
    HealthCheck
)
from yolo_model import infer, get_model_info, cleanup_model

# 설정 로드
settings = get_settings()

# 로깅 설정
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# 업로드된 원본+바운딩 박스 그린 이미지를 저장할 디렉토리
IMAGES_DIR = settings.IMAGES_DIR
os.makedirs(IMAGES_DIR, exist_ok=True)

# 지원하는 이미지 형식
SUPPORTED_FORMATS = set(settings.SUPPORTED_FORMATS)


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
    cleanup_model()
    logger.info("애플리케이션 종료 완료")


app = FastAPI(
    title=settings.APP_NAME,
    description="YOLO 기반 이미지 분석 API",
    version=settings.APP_VERSION,
    lifespan=lifespan  # lifespan 이벤트 핸들러 등록
)

# CORS 설정 최적화
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)


def validate_image_file(file: UploadFile) -> None:
    """이미지 파일 유효성 검사"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일명이 없습니다")

    file_ext = os.path.splitext(file.filename.lower())[1]
    if file_ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(SUPPORTED_FORMATS)}"
        )

    # 파일 크기 제한
    if file.size and file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"파일 크기가 너무 큽니다 (최대 {settings.MAX_FILE_SIZE // (1024 * 1024)}MB)"
        )


async def process_image_analysis(
        file_data: bytes,
        db: Session,
) -> List[ImageAnalysisBase]:
    """이미지 분석 처리 (백그라운드 작업용)"""
    try:
        # 1) 이미지 읽기
        img = read_image(file_data)

        # 2) 추론
        detections, annotated = infer(img)

        # 3) 주석된 이미지 파일로 저장
        annotated_image_path = save_annotated_image(annotated, prefix=f"{IMAGES_DIR}/")

        # 4) DB 저장 및 응답용 리스트 구성
        results = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            obj = crud.save_result(
                db,
                class_name=det["label"],
                confidence=det["confidence"],
                x1=x1, y1=y1, x2=x2, y2=y2,
                image_path=annotated_image_path
            )
            # Pydantic 모델로 변환하여 반환
            results.append(
                ImageAnalysisBase(
                    id=obj.id,
                    class_name=obj.class_name,
                    confidence=obj.confidence,
                    x1=obj.x1, y1=obj.y1,
                    x2=obj.x2, y2=obj.y2,
                    image_path=obj.image_path,
                    created_at=obj.created_at
                )
            )

        logger.info(f"이미지 분석 완료: {len(results)}개 객체 감지")
        return results

    except Exception as e:
        logger.error(f"이미지 분석 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail="이미지 분석 중 오류가 발생했습니다")


@app.post("/upload", response_model=ImageAnalysisList, responses={202: {"description": "Accepted", "model": dict}})
async def upload_image(
        file: UploadFile = File(...),
        db: Session = Depends(get_db),
        background_tasks: BackgroundTasks = None
):
    """
    이미지 업로드 및 분석

    - **file**: 분석할 이미지 파일 (JPG, PNG, BMP, TIFF 지원)
    - **background_tasks**: 백그라운드 작업 처리 (선택사항)
    """
    try:
        validate_image_file(file)
        data = await file.read()

        if background_tasks:  # 백그라운드 태스크 객체가 있다면 비동기로 처리
            background_tasks.add_task(process_image_analysis, data, db)
            return JSONResponse(
                content={"message": "이미지 분석이 백그라운드에서 시작되었습니다"},
                status_code=202
            )
        else:  # 백그라운드 태스크 객체가 없다면 동기로 처리 (주로 테스트 환경에서 사용)
            results = await process_image_analysis(data, db)
            return ImageAnalysisList(
                results=results,
                total_count=len(results),
                page=1,
                page_size=max(len(results), 1)
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"업로드 처리 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다")


@app.get("/analyses", response_model=ImageAnalysisList)
async def get_analyses(
        page: int = Query(1, ge=1, description="페이지 번호"),
        page_size: int = Query(100, ge=1, le=1000, description="페이지 크기"),
        class_name: Optional[str] = Query(None, description="클래스명으로 필터링"),
        db: Session = Depends(get_db)
):
    """
    분석 결과 목록 조회

    - **page**: 페이지 번호 (기본값: 1)
    - **page_size**: 페이지 크기 (기본값: 100, 최대: 1000)
    - **class_name**: 클래스명으로 필터링 (선택사항)
    """
    try:
        if class_name:
            analyses = crud.get_analyses_by_class(db, class_name, skip=(page - 1) * page_size, limit=page_size)
            total_count = crud.get_analyses_count_by_class(db, class_name)
        else:
            analyses = crud.get_recent_analyses(db, skip=(page - 1) * page_size, limit=page_size)
            total_count = crud.get_total_analyses_count(db)

        return ImageAnalysisList(
            results=analyses,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
    except Exception as e:
        logger.error(f"분석 결과 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail="분석 결과 조회 중 오류가 발생했습니다")


@app.get("/analyses/{analysis_id}", response_model=ImageAnalysisBase)
async def get_analysis(analysis_id: int, db: Session = Depends(get_db)):
    """
    특정 분석 결과 조회

    - **analysis_id**: 분석 결과 ID
    """
    try:
        analysis = crud.get_analysis_by_id(db, analysis_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="분석 결과를 찾을 수 없습니다")
        return analysis
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"분석 결과 조회 실패 (ID: {analysis_id}): {str(e)}")
        raise HTTPException(status_code=500, detail="분석 결과 조회 중 오류가 발생했습니다")


@app.delete("/analyses/{analysis_id}")
async def delete_analysis(analysis_id: int, db: Session = Depends(get_db)):
    """
    분석 결과 삭제

    - **analysis_id**: 삭제할 분석 결과 ID
    """
    try:
        success = crud.delete_analysis(db, analysis_id)
        if not success:
            raise HTTPException(status_code=404, detail="삭제할 분석 결과를 찾을 수 없습니다")
        return {"message": "분석 결과가 성공적으로 삭제되었습니다"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"분석 결과 삭제 실패 (ID: {analysis_id}): {str(e)}")
        raise HTTPException(status_code=500, detail="분석 결과 삭제 중 오류가 발생했습니다")


@app.get("/statistics", response_model=AnalysisStatistics)
async def get_statistics(db: Session = Depends(get_db)):
    """
    분석 통계 조회
    """
    try:
        stats = crud.get_statistics(db)
        return stats
    except Exception as e:
        logger.error(f"통계 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail="통계 조회 중 오류가 발생했습니다")


@app.get("/model/info")
async def get_model_info_endpoint():
    """
    YOLO 모델 정보 조회
    """
    try:
        info = get_model_info()
        return info
    except Exception as e:
        logger.error(f"모델 정보 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail="모델 정보 조회 중 오류가 발생했습니다")


@app.get("/health", response_model=HealthCheck)
async def health_check_endpoint():
    """헬스 체크 엔드포인트"""
    try:
        # 데이터베이스 연결 확인
        db_status = test_db_connection()

        return {
            "status": "healthy" if db_status else "unhealthy",
            "service": "image-analysis-api",
            "database": "connected" if db_status else "disconnected"
        }
    except Exception as e:
        logger.error(f"헬스 체크 실패: {str(e)}")
        raise HTTPException(status_code=500, detail="헬스 체크 중 오류가 발생했습니다")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """전역 예외 처리"""
    logger.error(f"예상치 못한 오류: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "서버 내부 오류가 발생했습니다"}
    )
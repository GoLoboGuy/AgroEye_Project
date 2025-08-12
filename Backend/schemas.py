from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import List, Optional
from enum import Enum


class AnalysisStatus(str, Enum):
    """분석 상태 열거형"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ImageAnalysisBase(BaseModel):
    """이미지 분석 결과 기본 스키마"""
    id: int = Field(..., description="분석 결과 ID")
    class_name: str = Field(..., min_length=1, max_length=50, description="감지된 객체 클래스명")
    confidence: float = Field(..., ge=0.0, le=1.0, description="신뢰도 (0.0-1.0)")
    x1: float = Field(..., description="바운딩 박스 좌상단 X 좌표")
    y1: float = Field(..., description="바운딩 박스 좌상단 Y 좌표")
    x2: float = Field(..., description="바운딩 박스 우하단 X 좌표")
    y2: float = Field(..., description="바운딩 박스 우하단 Y 좌표")
    image_path: str = Field(..., max_length=200, description="저장된 이미지 파일 경로")
    created_at: datetime = Field(..., description="생성 시간")

    # Pydantic V2의 field_validator를 사용하여 유효성 검사
    @field_validator('confidence', mode='before')
    @classmethod
    def validate_confidence(cls, v):
        """신뢰도 검증"""
        if not 0.0 <= v <= 1.0:
            raise ValueError('신뢰도는 0.0과 1.0 사이여야 합니다')
        return v

    @field_validator('x1', 'y1', 'x2', 'y2', mode='before')
    @classmethod
    def validate_coordinates(cls, v):
        """좌표 검증"""
        if v < 0:
            raise ValueError('좌표는 음수일 수 없습니다')
        return v

    @field_validator('x2')
    @classmethod
    def validate_x2(cls, v, info):
        """X2 좌표 검증"""
        # info.data를 사용하여 다른 필드 값에 접근
        if 'x1' in info.data and v <= info.data['x1']:
            raise ValueError('x2는 x1보다 커야 합니다')
        return v

    @field_validator('y2')
    @classmethod
    def validate_y2(cls, v, info):
        """Y2 좌표 검증"""
        # info.data를 사용하여 다른 필드 값에 접근
        if 'y1' in info.data and v <= info.data['y1']:
            raise ValueError('y2는 y1보다 커야 합니다')
        return v

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "example": {
                "id": 1,
                "class_name": "person",
                "confidence": 0.95,
                "x1": 100.0,
                "y1": 150.0,
                "x2": 300.0,
                "y2": 450.0,
                "image_path": "images/abc123.jpg",
                "created_at": "2024-01-01T12:00:00Z"
            }
        }
    }


class ImageAnalysisCreate(BaseModel):
    """이미지 분석 생성 스키마"""
    class_name: str = Field(..., min_length=1, max_length=50)
    confidence: float = Field(..., ge=0.0, le=1.0)
    x1: float = Field(..., ge=0.0)
    y1: float = Field(..., ge=0.0)
    x2: float = Field(..., ge=0.0)
    y2: float = Field(..., ge=0.0)
    image_path: str = Field(..., max_length=200)

    # x2, y2 필드 간의 유효성 검사
    @field_validator('x2')
    @classmethod
    def validate_x2_create(cls, v, info):
        if 'x1' in info.data and v <= info.data['x1']:
            raise ValueError('x2는 x1보다 커야 합니다')
        return v

    @field_validator('y2')
    @classmethod
    def validate_y2_create(cls, v, info):
        if 'y1' in info.data and v <= info.data['y1']:
            raise ValueError('y2는 y1보다 커야 합니다')
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "class_name": "person",
                "confidence": 0.95,
                "x1": 100.0,
                "y1": 150.0,
                "x2": 300.0,
                "y2": 450.0,
                "image_path": "images/abc123.jpg"
            }
        }
    }


class ImageAnalysisUpdate(BaseModel):
    """이미지 분석 업데이트 스키마"""
    class_name: Optional[str] = Field(None, min_length=1, max_length=50)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    x1: Optional[float] = Field(None, ge=0.0)
    y1: Optional[float] = Field(None, ge=0.0)
    x2: Optional[float] = Field(None, ge=0.0)
    y2: Optional[float] = Field(None, ge=0.0)
    image_path: Optional[str] = Field(None, max_length=200)

    # x2, y2 필드 간의 유효성 검사
    @field_validator('x2')
    @classmethod
    def validate_x2_update(cls, v, info):
        if 'x1' in info.data and v is not None and v <= info.data['x1']:
            raise ValueError('x2는 x1보다 커야 합니다')
        return v

    @field_validator('y2')
    @classmethod
    def validate_y2_update(cls, v, info):
        if 'y1' in info.data and v is not None and v <= info.data['y1']:
            raise ValueError('y2는 y1보다 커야 합니다')
        return v


class ImageAnalysisList(BaseModel):
    """이미지 분석 결과 목록 스키마"""
    results: List[ImageAnalysisBase] = Field(..., description="분석 결과 목록")
    total_count: int = Field(..., description="총 결과 수")
    page: Optional[int] = Field(1, ge=1, description="현재 페이지")
    page_size: Optional[int] = Field(100, ge=1, le=1000, description="페이지 크기")

    model_config = {
        "json_schema_extra": {
            "example": {
                "results": [
                    {
                        "id": 1,
                        "class_name": "person",
                        "confidence": 0.95,
                        "x1": 100.0,
                        "y1": 150.0,
                        "x2": 300.0,
                        "y2": 450.0,
                        "image_path": "images/abc123.jpg",
                        "created_at": "2024-01-01T12:00:00Z"
                    }
                ],
                "total_count": 1,
                "page": 1,
                "page_size": 100
            }
        }
    }


class AnalysisStatistics(BaseModel):
    """분석 통계 스키마"""
    total_analyses: int = Field(..., description="총 분석 수")
    class_statistics: List[dict] = Field(..., description="클래스별 통계")

    model_config = {
        "json_schema_extra": {
            "example": {
                "total_analyses": 100,
                "class_statistics": [
                    {
                        "class_name": "person",
                        "count": 50,
                        "avg_confidence": 0.85
                    },
                    {
                        "class_name": "car",
                        "count": 30,
                        "avg_confidence": 0.92
                    }
                ]
            }
        }
    }


class HealthCheck(BaseModel):
    """헬스 체크 응답 스키마"""
    status: str = Field(..., description="서비스 상태")
    service: str = Field(..., description="서비스명")
    timestamp: datetime = Field(default_factory=datetime.now, description="체크 시간")
    version: str = Field("1.0.0", description="API 버전")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "service": "image-analysis-api",
                "timestamp": "2024-01-01T12:00:00Z",
                "version": "1.0.0"
            }
        }
    }


class ErrorResponse(BaseModel):
    """에러 응답 스키마"""
    detail: str = Field(..., description="에러 메시지")
    error_code: Optional[str] = Field(None, description="에러 코드")
    timestamp: datetime = Field(default_factory=datetime.now, description="에러 발생 시간")

    model_config = {
        "json_schema_extra": {
            "example": {
                "detail": "이미지 파일을 읽을 수 없습니다",
                "error_code": "INVALID_IMAGE",
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }
    }
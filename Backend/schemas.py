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


class finalprojectresultsbase(BaseModel):
    """최종 프로젝트 결과 기본 스키마"""
    id: int = Field(..., description="결과 ID")
    class_name: str = Field(..., min_length=1, max_length=255, description="예측된 클래스명")
    class_info: Optional[str] = Field(None, description="클래스 정보")
    recomm: Optional[str] = Field(None, description="추천사항")
    image_path: str = Field(..., max_length=500, description="이미지 파일 경로")
    created_at: datetime = Field(..., description="생성 시간")
    updated_at: datetime = Field(..., description="수정 시간")

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "example": {
                "id": 1,
                "class_name": "healthy_leaf",
                "class_info": "건강한 잎사귀입니다",
                "recomm": "정기적인 관수를 유지하세요",
                "image_path": "images/leaf_001.jpg",
                "created_at": "2024-01-01T12:00:00Z",
                "updated_at": "2024-01-01T12:00:00Z"
            }
        }
    }


class finalprojectresultscreate(BaseModel):
    """최종 프로젝트 결과 생성 스키마"""
    class_name: str = Field(..., min_length=1, max_length=255, description="예측된 클래스명")
    class_info: Optional[str] = Field(None, description="클래스 정보")
    recomm: Optional[str] = Field(None, description="추천사항")
    image_path: str = Field(..., max_length=500, description="이미지 파일 경로")

    model_config = {
        "json_schema_extra": {
            "example": {
                "class_name": "healthy_leaf",
                "class_info": "건강한 잎사귀입니다",
                "recomm": "정기적인 관수를 유지하세요",
                "image_path": "images/leaf_001.jpg"
            }
        }
    }


class finalprojectresultsupdate(BaseModel):
    """최종 프로젝트 결과 업데이트 스키마"""
    class_name: Optional[str] = Field(None, min_length=1, max_length=255)
    class_info: Optional[str] = Field(None)
    recomm: Optional[str] = Field(None)
    image_path: Optional[str] = Field(None, max_length=500)


class finalprojectresultslist(BaseModel):
    """최종 프로젝트 결과 목록 스키마"""
    results: List[finalprojectresultsbase] = Field(..., description="결과 목록")
    total_count: int = Field(..., description="총 결과 수")
    page: Optional[int] = Field(1, ge=1, description="현재 페이지")
    page_size: Optional[int] = Field(100, ge=1, le=1000, description="페이지 크기")

    model_config = {
        "json_schema_extra": {
            "example": {
                "results": [
                    {
                        "id": 1,
                        "class_name": "healthy_leaf",
                        "class_info": "건강한 잎사귀입니다",
                        "recomm": "정기적인 관수를 유지하세요",
                        "image_path": "images/leaf_001.jpg",
                        "created_at": "2024-01-01T12:00:00Z",
                        "updated_at": "2024-01-01T12:00:00Z"
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
                        "class_name": "healthy_leaf",
                        "count": 50
                    },
                    {
                        "class_name": "diseased_leaf",
                        "count": 30
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
                "service": "leaf-disease-detection-api",
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
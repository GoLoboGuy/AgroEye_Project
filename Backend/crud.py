# backend/crud.py
import logging
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc
from models import ImageAnalysis

logger = logging.getLogger(__name__)

def save_result(db: Session, *, class_name: str, confidence: float,
                x1: float, y1: float, x2: float, y2: float,
                image_path: str) -> ImageAnalysis:
    """
    단일 분석 결과를 저장합니다.
    """
    try:
        obj = ImageAnalysis(
            class_name=class_name,
            confidence=confidence,
            x1=x1, y1=y1, x2=x2, y2=y2,
            image_path=image_path
        )
        db.add(obj)
        db.commit()
        db.refresh(obj)
        logger.info(f"분석 결과 저장 완료: {class_name} (신뢰도: {confidence:.2f})")
        return obj
    except Exception as e:
        logger.error(f"분석 결과 저장 실패: {str(e)}")
        db.rollback()
        raise

def save_results_batch(db: Session, results: List[dict]) -> List[ImageAnalysis]:
    """
    여러 분석 결과를 배치로 저장합니다.
    """
    try:
        # bulk_insert_mappings 사용으로 대량 삽입 최적화
        db.bulk_insert_mappings(ImageAnalysis, results)
        db.commit()
        # 삽입된 레코드 조회
        inserted = (
            db.query(ImageAnalysis)
            .filter(ImageAnalysis.image_path == results[0]["image_path"])
            .order_by(ImageAnalysis.created_at.desc())
            .limit(len(results))
            .all()
        )
        logger.info(f"배치 저장 완료: {len(inserted)}개 결과")
        return inserted
    except Exception as e:
        logger.error(f"배치 저장 실패: {str(e)}")
        db.rollback()
        raise

def get_analysis_by_id(db: Session, analysis_id: int) -> Optional[ImageAnalysis]:
    """
    ID로 분석 결과를 조회합니다.
    """
    try:
        return db.query(ImageAnalysis).filter(ImageAnalysis.id == analysis_id).first()
    except Exception as e:
        logger.error(f"분석 결과 조회 실패 (ID: {analysis_id}): {str(e)}")
        raise

def get_analyses_by_image_path(db: Session, image_path: str) -> List[ImageAnalysis]:
    """
    이미지 경로로 분석 결과들을 조회합니다.
    """
    try:
        return db.query(ImageAnalysis).filter(
            ImageAnalysis.image_path == image_path
        ).order_by(desc(ImageAnalysis.created_at)).all()
    except Exception as e:
        logger.error(f"이미지 경로로 조회 실패 ({image_path}): {str(e)}")
        raise

def get_recent_analyses(db: Session, limit: int = 100) -> List[ImageAnalysis]:
    """
    최근 분석 결과들을 조회합니다.
    """
    try:
        return db.query(ImageAnalysis).order_by(
            desc(ImageAnalysis.created_at)
        ).limit(limit).all()
    except Exception as e:
        logger.error(f"최근 분석 결과 조회 실패: {str(e)}")
        raise

def get_analyses_by_class(db: Session, class_name: str, limit: int = 100) -> List[ImageAnalysis]:
    """
    특정 클래스의 분석 결과들을 조회합니다.
    """
    try:
        return db.query(ImageAnalysis).filter(
            ImageAnalysis.class_name == class_name
        ).order_by(desc(ImageAnalysis.created_at)).limit(limit).all()
    except Exception as e:
        logger.error(f"클래스별 분석 결과 조회 실패 ({class_name}): {str(e)}")
        raise

def delete_analysis(db: Session, analysis_id: int) -> bool:
    """
    분석 결과를 삭제합니다.
    """
    try:
        analysis = db.query(ImageAnalysis).filter(ImageAnalysis.id == analysis_id).first()
        if analysis:
            db.delete(analysis)
            db.commit()
            logger.info(f"분석 결과 삭제 완료 (ID: {analysis_id})")
            return True
        else:
            logger.warning(f"삭제할 분석 결과를 찾을 수 없습니다 (ID: {analysis_id})")
            return False
    except Exception as e:
        logger.error(f"분석 결과 삭제 실패 (ID: {analysis_id}): {str(e)}")
        db.rollback()
        raise

def get_statistics(db: Session) -> dict:
    """
    분석 통계를 조회합니다.
    """
    try:
        total_count = db.query(ImageAnalysis).count()
        
        # 클래스별 통계
        class_stats = db.query(
            ImageAnalysis.class_name,
            db.func.count(ImageAnalysis.id).label('count'),
            db.func.avg(ImageAnalysis.confidence).label('avg_confidence')
        ).group_by(ImageAnalysis.class_name).all()
        
        return {
            "total_analyses": total_count,
            "class_statistics": [
                {
                    "class_name": stat.class_name,
                    "count": stat.count,
                    "avg_confidence": float(stat.avg_confidence) if stat.avg_confidence else 0.0
                }
                for stat in class_stats
            ]
        }
    except Exception as e:
        logger.error(f"통계 조회 실패: {str(e)}")
        raise

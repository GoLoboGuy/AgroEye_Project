# backend/crud.py
import logging
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc
from models import finalprojectresults

logger = logging.getLogger(__name__)

def save_result(db: Session, *, class_name: str, class_info: Optional[str] = None,
                recomm: Optional[str] = None, image_path: str) -> finalprojectresults:
    """
    단일 예측 결과를 저장합니다.
    """
    try:
        logger.info(f"예측 결과 저장 시작: class_name={class_name}, image_path={image_path}")
        
        # 입력 데이터 검증
        if not class_name:
            raise ValueError("class_name은 필수입니다")
        if not image_path:
            raise ValueError("image_path는 필수입니다")
            
        logger.info(f"데이터 검증 완료, 객체 생성 중...")
        
        # 객체 생성
        obj = finalprojectresults(
            class_name=class_name,
            class_info=class_info,
            recomm=recomm,
            image_path=image_path
        )
        
        logger.info(f"객체 생성 완료: {obj}")
        
        # 데이터베이스에 추가
        logger.info("데이터베이스에 객체 추가 중...")
        db.add(obj)
        
        # 커밋
        logger.info("데이터베이스 커밋 중...")
        db.commit()
        
        # 객체 새로고침
        logger.info("객체 새로고침 중...")
        db.refresh(obj)
        
        logger.info(f"예측 결과 저장 완료: ID={obj.id}, class_name={class_name}")
        return obj
        
    except Exception as e:
        logger.error(f"예측 결과 저장 실패: {str(e)}")
        logger.error(f"에러 타입: {type(e).__name__}")
        logger.error(f"입력 데이터: class_name={class_name}, class_info={class_info}, recomm={recomm}, image_path={image_path}")
        
        # 데이터베이스 상태 확인
        try:
            logger.error(f"데이터베이스 상태: {db.is_active}")
        except:
            logger.error("데이터베이스 상태 확인 불가")
        
        # 상세 에러 정보
        import traceback
        logger.error(f"상세 에러: {traceback.format_exc()}")
        
        # 롤백
        try:
            db.rollback()
            logger.info("데이터베이스 롤백 완료")
        except Exception as rollback_error:
            logger.error(f"롤백 실패: {str(rollback_error)}")
        
        raise

def save_results_batch(db: Session, results: List[dict]) -> List[finalprojectresults]:
    """
    여러 예측 결과를 배치로 저장합니다.
    """
    try:
        # bulk_insert_mappings 사용으로 대량 삽입 최적화
        db.bulk_insert_mappings(finalprojectresults, results)
        db.commit()
        # 삽입된 레코드 조회
        inserted = (
            db.query(finalprojectresults)
            .filter(finalprojectresults.image_path == results[0]["image_path"])
            .order_by(finalprojectresults.created_at.desc())
            .limit(len(results))
            .all()
        )
        logger.info(f"배치 저장 완료: {len(inserted)}개 결과")
        return inserted
    except Exception as e:
        logger.error(f"배치 저장 실패: {str(e)}")
        db.rollback()
        raise

def get_result_by_id(db: Session, result_id: int) -> Optional[finalprojectresults]:
    """
    ID로 예측 결과를 조회합니다.
    """
    try:
        return db.query(finalprojectresults).filter(finalprojectresults.id == result_id).first()
    except Exception as e:
        logger.error(f"예측 결과 조회 실패 (ID: {result_id}): {str(e)}")
        raise

def get_results_by_image_path(db: Session, image_path: str) -> List[finalprojectresults]:
    """
    이미지 경로로 예측 결과들을 조회합니다.
    """
    try:
        return db.query(finalprojectresults).filter(
            finalprojectresults.image_path == image_path
        ).order_by(desc(finalprojectresults.created_at)).all()
    except Exception as e:
        logger.error(f"이미지 경로로 조회 실패 ({image_path}): {str(e)}")
        raise

def get_recent_results(db: Session, limit: int = 100) -> List[finalprojectresults]:
    """
    최근 예측 결과들을 조회합니다.
    """
    try:
        return db.query(finalprojectresults).order_by(
            desc(finalprojectresults.created_at)
        ).limit(limit).all()
    except Exception as e:
        logger.error(f"최근 예측 결과 조회 실패: {str(e)}")
        raise

def get_results_by_class(db: Session, class_name: str, limit: int = 100) -> List[finalprojectresults]:
    """
    특정 클래스의 예측 결과들을 조회합니다.
    """
    try:
        return db.query(finalprojectresults).filter(
            finalprojectresults.class_name == class_name
        ).order_by(desc(finalprojectresults.created_at)).limit(limit).all()
    except Exception as e:
        logger.error(f"클래스별 예측 결과 조회 실패 ({class_name}): {str(e)}")
        raise

def delete_result(db: Session, result_id: int) -> bool:
    """
    예측 결과를 삭제합니다.
    """
    try:
        result = db.query(finalprojectresults).filter(finalprojectresults.id == result_id).first()
        if result:
            db.delete(result)
            db.commit()
            logger.info(f"예측 결과 삭제 완료 (ID: {result_id})")
            return True
        else:
            logger.warning(f"삭제할 예측 결과를 찾을 수 없습니다 (ID: {result_id})")
            return False
    except Exception as e:
        logger.error(f"예측 결과 삭제 실패 (ID: {result_id}): {str(e)}")
        db.rollback()
        raise

def get_statistics(db: Session) -> dict:
    """
    예측 통계를 조회합니다.
    """
    try:
        total_count = db.query(finalprojectresults).count()
        
        # 클래스별 통계
        class_stats = db.query(
            finalprojectresults.class_name,
            db.func.count(finalprojectresults.id).label('count')
        ).group_by(finalprojectresults.class_name).all()
        
        return {
            "total_analyses": total_count,
            "class_statistics": [
                {
                    "class_name": stat.class_name,
                    "count": stat.count
                }
                for stat in class_stats
            ]
        }
    except Exception as e:
        logger.error(f"통계 조회 실패: {str(e)}")
        raise

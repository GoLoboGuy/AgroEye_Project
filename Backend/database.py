# backend/database.py
import logging
import os
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from config import get_settings
from models import Base

# 설정 로드
settings = get_settings()

# 로깅 설정
logger = logging.getLogger(__name__)

# 환경변수 로드
load_dotenv()

# 데이터베이스 URL 설정 (기본값: SQLite)
DATABASE_URL = os.getenv("DATABASE_URL", settings.DATABASE_URL)

logger.info(f"데이터베이스 URL: {DATABASE_URL}")

# 엔진 설정 최적화
engine = create_engine(
    DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
    poolclass=QueuePool,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_timeout=30,           # 대기 시간 (초)
    pool_pre_ping=True,  # 연결 상태 확인
    pool_recycle=settings.DB_POOL_RECYCLE,
)

# 세션 팩토리 설정
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False  # 커밋 후 객체 만료 방지
)

# 데이터베이스 연결 테스트
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """SQLite 성능 최적화 설정"""
    if "sqlite" in DATABASE_URL:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=10000")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.close()

def init_db():
    """데이터베이스 초기화"""
    try:
        logger.info("데이터베이스 초기화 시작...")
        
        # 테이블이 존재하지 않으면 자동 생성
        Base.metadata.create_all(bind=engine)
        logger.info("SQLAlchemy 테이블 생성 완료")
        
        # MySQL의 경우 테이블 존재 여부 확인
        if "mysql" in DATABASE_URL.lower():
            logger.info("MySQL 데이터베이스 감지, 테이블 확인 중...")
            with engine.connect() as connection:
                # final_project_results 테이블 존재 여부 확인
                result = connection.execute(text("""
                    SELECT COUNT(*) 
                    FROM information_schema.tables 
                    WHERE table_schema = DATABASE() 
                    AND table_name = 'final_project_results'
                """))
                table_exists = result.scalar() > 0
                
                if not table_exists:
                    logger.info("final_project_results 테이블이 존재하지 않습니다. 생성 중...")
                    # 테이블이 없으면 생성
                    connection.execute(text("""
                        CREATE TABLE IF NOT EXISTS final_project_results (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            class_name VARCHAR(255) NOT NULL,
                            class_info TEXT NULL,
                            recomm TEXT NULL,
                            image_path VARCHAR(500) NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                            INDEX idx_class_name (class_name),
                            INDEX idx_created_at (created_at)
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                    """))
                    connection.commit()
                    logger.info("final_project_results 테이블이 성공적으로 생성되었습니다.")
                else:
                    logger.info("final_project_results 테이블이 이미 존재합니다.")
                    
        elif "sqlite" in DATABASE_URL.lower():
            logger.info("SQLite 데이터베이스 감지")
        else:
            logger.info(f"데이터베이스 타입: {DATABASE_URL}")
            
        logger.info("데이터베이스 초기화가 성공적으로 완료되었습니다.")
        
    except Exception as e:
        logger.error(f"데이터베이스 초기화 중 오류 발생: {str(e)}")
        logger.error(f"오류 타입: {type(e).__name__}")
        import traceback
        logger.error(f"상세 오류: {traceback.format_exc()}")
        raise

def get_db():
    """데이터베이스 세션 의존성"""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"데이터베이스 세션 오류: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()

@contextmanager
def get_db_context():
    """컨텍스트 매니저를 사용한 데이터베이스 세션"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        logger.error(f"데이터베이스 컨텍스트 오류: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()

def test_db_connection():
    """데이터베이스 연결 테스트"""
    try:
        with engine.connect() as connection:
            # raw SQL 은 text() 로 래핑해야 실행 가능합니다.
            connection.execute(text("SELECT 1"))
        logger.info("데이터베이스 연결이 정상입니다.")
        return True
    except Exception as e:
        logger.error(f"데이터베이스 연결 테스트 실패: {str(e)}")
        return False

# 서버 시작 시 데이터베이스 초기화
if __name__ == "__main__":
    init_db()
    test_db_connection()

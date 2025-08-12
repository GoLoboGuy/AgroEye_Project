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

# 엔진 설정 최적화
engine = create_engine(
    DATABASE_URL,
    echo=False,  # 프로덕션에서는 False
    future=True,
    poolclass=QueuePool,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
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
        Base.metadata.create_all(bind=engine)
        logger.info("데이터베이스 테이블이 성공적으로 생성되었습니다.")
    except Exception as e:
        logger.error(f"데이터베이스 초기화 중 오류 발생: {str(e)}")
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

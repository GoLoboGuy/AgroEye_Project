# Backend

FastAPI 서버, SQLAlchemy + MySQL 연동, Alembic 마이그레이션, YOLO 추론을 포함합니다.

## 설정

1. `.env` 파일에 DB_URL 설정
2. Conda 환경 생성 및 활성화
   ```bash
   conda env create -f environment.yml
   conda activate agroeye
   ```
3. 마이그레이션
   ```bash
   alembic upgrade head
   ```
4. 서버 실행
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

또는 Docker:
```bash
docker build -t agroeye-backend .
docker run --env-file .env -p 8000:8000 agroeye-backend
```

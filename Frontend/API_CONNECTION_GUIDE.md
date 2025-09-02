# 🔗 API 연결 설정 가이드

이 가이드는 Frontend와 Backend를 서로 다른 컴퓨터에서 실행할 때 API 연결을 설정하는 방법을 설명합니다.

## 🎯 **설정 방법**

### **1. 환경 변수 설정**

Frontend 루트 디렉토리에 `.env` 파일을 생성하고 다음 내용을 추가하세요:

```bash
# .env 파일
VITE_API_BASE_URL=/api
VITE_PROXY_TARGET=http://BACKEND_SERVER_IP:8000
VITE_DEBUG=true
```

**예시:**
```bash
# Backend 서버가 192.168.1.100에서 실행 중인 경우
VITE_API_BASE_URL=/api
VITE_PROXY_TARGET=http://192.168.1.100:8000
VITE_DEBUG=true
```

### **2. Backend 서버 실행**

Backend 컴퓨터에서 다음 명령으로 서버를 실행하세요:

```bash
cd Backend
uvicorn Backend.main:app --host 0.0.0.0 --port 8000
```

⚠️ **중요:** `--host 0.0.0.0`을 반드시 포함해야 다른 컴퓨터에서 접근할 수 있습니다.

### **3. 방화벽 설정**

Backend 컴퓨터에서 8000 포트를 허용하세요:

**Windows:**
1. Windows Defender 방화벽 열기
2. "고급 설정" 클릭
3. "인바운드 규칙" → "새 규칙"
4. "포트" 선택 → "TCP" → "특정 로컬 포트" → "8000" 입력

**Linux:**
```bash
sudo ufw allow 8000
```

### **4. Frontend 실행**

Frontend 컴퓨터에서 다음 명령으로 실행하세요:

```bash
cd Frontend
npm run dev
```

## 🔍 **연결 상태 확인**

Frontend 실행 후 웹페이지 상단에 API 연결 상태가 표시됩니다:

- 🟢 **녹색**: Backend 서버 연결됨
- 🔴 **빨간색**: Backend 서버 연결 실패
- 🟡 **노란색**: 연결 확인 중

연결 상태 표시기를 클릭하면 상세 정보를 볼 수 있습니다.

## 🛠️ **문제 해결**

### **연결이 안 될 때:**

1. **Backend 서버 확인:**
   ```bash
   # Backend 컴퓨터에서 확인
   curl http://localhost:8000/health
   ```

2. **네트워크 연결 확인:**
   ```bash
   # Frontend 컴퓨터에서 확인
   curl http://BACKEND_SERVER_IP:8000/health
   ```

3. **방화벽 확인:**
   - Backend 컴퓨터의 8000 포트가 열려있는지 확인
   - 라우터/공유기의 포트 포워딩 설정 확인

4. **IP 주소 확인:**
   ```bash
   # Backend 컴퓨터에서 IP 확인
   ipconfig    # Windows
   ifconfig    # Linux/Mac
   ```

### **프록시 vs 직접 연결:**

**프록시 사용 (권장):**
```bash
VITE_API_BASE_URL=/api
VITE_PROXY_TARGET=http://BACKEND_SERVER_IP:8000
```

**직접 연결:**
```bash
VITE_API_BASE_URL=http://BACKEND_SERVER_IP:8000/api
# VITE_PROXY_TARGET은 설정하지 않음
```

## 📱 **다양한 환경 설정**

### **개발 환경별 설정:**

**.env.development (개발용):**
```bash
VITE_API_BASE_URL=/api
VITE_PROXY_TARGET=http://localhost:8000
VITE_DEBUG=true
```

**.env.production (운영용):**
```bash
VITE_API_BASE_URL=/api
VITE_PROXY_TARGET=http://production-server:8000
VITE_DEBUG=false
```

### **팀 개발 시:**

각 개발자는 자신의 `.env` 파일에서 `VITE_PROXY_TARGET`만 수정하면 됩니다:

```bash
# 개발자 A
VITE_PROXY_TARGET=http://192.168.1.100:8000

# 개발자 B  
VITE_PROXY_TARGET=http://192.168.1.200:8000
```

## 🔧 **고급 설정**

### **디버그 모드:**
```bash
VITE_DEBUG=true
```
디버그 모드를 활성화하면 브라우저 콘솔에서 프록시 요청/응답 로그를 확인할 수 있습니다.

### **타임아웃 설정:**
필요시 `Frontend/src/services/api.ts`에서 타임아웃 값을 조정할 수 있습니다:

```typescript
export const backendApi = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30초 (필요에 따라 조정)
  // ...
})
```

## 🎉 **성공 확인**

모든 설정이 완료되면:

1. Frontend 페이지 상단에 녹색 연결 상태 표시
2. 카메라로 이미지 촬영 및 AI 분석 정상 작동
3. 브라우저 개발자 도구의 Network 탭에서 API 요청 확인 가능

---

문제가 지속되면 브라우저 개발자 도구의 Console과 Network 탭을 확인하여 구체적인 오류 메시지를 찾아보세요.

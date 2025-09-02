import axios from 'axios'

// 환경 변수에서 API URL 읽기 (프록시 사용 시 '/api', 직접 연결 시 전체 URL)
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api'

// 디버그 모드 확인
const DEBUG_MODE = import.meta.env.VITE_DEBUG === 'true'

// Backend API 클라이언트
export const backendApi = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: false, // CORS 문제 방지
})

// API 연결 상태 확인을 위한 헬퍼 함수 (개발자용, 콘솔에서만 표시)
export const checkApiConnection = async (): Promise<boolean> => {
  try {
    const healthUrl = API_BASE_URL.startsWith('/')
      ? '/health'  // 프록시 사용 시
      : API_BASE_URL.replace('/api', '/health')  // 직접 연결 시

    const response = await axios.get(healthUrl, { timeout: 5000 })

    if (DEBUG_MODE) {
      console.log('✅ API 연결 성공:', healthUrl)
    }
    return response.status === 200
  } catch (error) {
    if (DEBUG_MODE) {
      console.error('❌ API 연결 실패:', error)
      console.log('🔧 디버깅 정보:')
      console.log('   - API URL:', API_BASE_URL)
      console.log('   - 프록시 사용:', API_BASE_URL.startsWith('/'))
      console.log('   - 환경 변수 VITE_DEBUG:', DEBUG_MODE)
    }
    return false
  }
}

// 현재 사용 중인 API URL 확인용 함수 (개발자용)
export const getCurrentApiUrl = (): string => {
  return API_BASE_URL
}

// 현재 프록시 사용 여부 확인 (개발자용)
export const isUsingProxy = (): boolean => {
  return API_BASE_URL.startsWith('/')
}

// 응답 인터셉터 추가 (에러 처리)
backendApi.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      // Backend에서 반환한 에러
      const apiError: ApiError = {
        detail: error.response.data.detail || '서버 오류가 발생했습니다.',
        status_code: error.response.status
      }
      throw apiError
    } else if (error.request) {
      // 네트워크 에러
      throw {
        detail: '네트워크 연결을 확인해주세요.',
        status_code: 0
      } as ApiError
    } else {
      // 기타 에러
      throw {
        detail: error.message || '알 수 없는 오류가 발생했습니다.',
        status_code: 0
      } as ApiError
    }
  }
)

// 에러 타입 정의
export interface ApiError {
  detail: string
  status_code?: number
}

// Backend API 응답 타입들
export interface SourceItem {
  source: string
  page?: number
  score?: number
  snippet?: string
  title?: string
}

export interface PredictResponse {
  id: number
  class_name: string
  confidence: number
  recomm: string
  image_path: string
  sources: SourceItem[]
  detailed_prediction?: Record<string, unknown>
}

export interface ResultItem {
  id: number
  class_name: string
  image_path: string
  created_at: string
}

export interface ResultsPage {
  total: number
  page: number
  size: number
  items: ResultItem[]
}

export interface ResultDetail {
  id: number
  class_name: string
  recomm: string
  image_path: string
  created_at: string
  updated_at: string
  class_info?: Record<string, unknown>
}

export interface DeleteResult {
  id: number
  deleted: boolean
}

export interface ModelStatus {
  model_loaded: boolean
  model_available: boolean
  status: string
}

export interface HealthCheck {
  status: string
  service: string
  timestamp: string
  version: string
}

// API 함수들
export const apiService = {
  // 이미지 업로드 및 예측
  async predictImage(imageFile: File): Promise<PredictResponse> {
    try {
      const formData = new FormData()
      formData.append('file', imageFile)

      const response = await backendApi.post<PredictResponse>('/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      return response.data
    } catch (error) {
      throw error // 인터셉터에서 처리된 에러를 다시 throw
    }
  },

  // 모델 상태 확인
  async getModelStatus(): Promise<ModelStatus> {
    try {
      const response = await backendApi.get<ModelStatus>('/model/status')
      return response.data
    } catch (error) {
      throw error
    }
  },

  // 결과 목록 조회 (페이지네이션 + 필터링)
  async getResults(
    page: number = 1,
    size: number = 20,
    class_name?: string,
    order: 'asc' | 'desc' = 'desc'
  ): Promise<ResultsPage> {
    try {
      const params: Record<string, string | number> = { page, size, order }
      if (class_name) {
        params.class_name = class_name
      }

      const response = await backendApi.get<ResultsPage>('/results', { params })
      return response.data
    } catch (error) {
      throw error
    }
  },

  // 특정 결과 상세 조회
  async getResultById(id: number): Promise<ResultDetail> {
    try {
      const response = await backendApi.get<ResultDetail>(`/results/${id}`)
      return response.data
    } catch (error) {
      throw error
    }
  },

  // 결과 삭제
  async deleteResult(id: number): Promise<DeleteResult> {
    try {
      const response = await backendApi.delete<DeleteResult>(`/results/${id}`)
      return response.data
    } catch (error) {
      throw error
    }
  },

  // 헬스 체크
  async healthCheck(): Promise<HealthCheck> {
    try {
      const healthUrl = API_BASE_URL.startsWith('/')
        ? '/health'  // 프록시 사용 시
        : API_BASE_URL.replace('/api', '/health')  // 직접 연결 시

      const response = await axios.get<HealthCheck>(healthUrl)
      return response.data
    } catch (error) {
      throw error
    }
  }
}

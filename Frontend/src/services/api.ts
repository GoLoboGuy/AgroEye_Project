import axios from 'axios'

// API 기본 설정
const API_BASE_URL = 'http://localhost:8000'
const MODEL_API_BASE_URL = 'http://localhost:8001'

// Backend API 클라이언트
export const backendApi = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Model API 클라이언트
export const modelApi = axios.create({
  baseURL: MODEL_API_BASE_URL,
  timeout: 60000, // 모델 추론은 더 오래 걸릴 수 있음
  headers: {
    'Content-Type': 'multipart/form-data',
  },
})

// 이미지 분석 결과 타입
export interface AnalysisResult {
  id: number
  class_name: string
  confidence: number
  x1: number
  y1: number
  x2: number
  y2: number
  image_path: string
  created_at: string
}

// 모델 예측 결과 타입
export interface ModelPrediction {
  mobilenet: {
    label: string
    confidence: number
  }
  resnet50: {
    label: string
    confidence: number
  }
  picked: {
    model: string
    label: string
    confidence: number
  }
}

// 병충해 정보 타입
export interface DiseaseInfo {
  disease_name: string
  description: string
  symptoms: string[]
  treatment: string[]
  prevention: string[]
  severity: 'low' | 'medium' | 'high'
}

// API 함수들
export const apiService = {
  // 이미지 업로드 및 분석
  async uploadAndAnalyzeImage(imageFile: File): Promise<AnalysisResult[]> {
    const formData = new FormData()
    formData.append('file', imageFile)

    const response = await backendApi.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })

    return response.data.results
  },

  // 모델 예측
  async predictImage(imageFile: File): Promise<ModelPrediction> {
    const formData = new FormData()
    formData.append('files', imageFile)

    const response = await modelApi.post('/api/predict', formData)
    return response.data.results[0]
  },

  // 분석 결과 목록 조회
  async getAnalysisResults(page: number = 1, pageSize: number = 100): Promise<{
    results: AnalysisResult[]
    total_count: number
    page: number
    page_size: number
  }> {
    const response = await backendApi.get('/analyses', {
      params: { page, page_size: pageSize }
    })
    return response.data
  },

  // 특정 분석 결과 조회
  async getAnalysisById(id: number): Promise<AnalysisResult> {
    const response = await backendApi.get(`/analyses/${id}`)
    return response.data
  },

  // 통계 정보 조회
  async getStatistics(): Promise<{
    total_analyses: number
    class_statistics: Array<{
      class_name: string
      count: number
      avg_confidence: number
    }>
  }> {
    const response = await backendApi.get('/statistics')
    return response.data
  },

  // 헬스 체크
  async healthCheck(): Promise<{
    status: string
    service: string
    database: string
  }> {
    const response = await backendApi.get('/health')
    return response.data
  },

  // 모델 정보 조회
  async getModelInfo(): Promise<any> {
    const response = await backendApi.get('/model/info')
    return response.data
  }
}

// 병충해 정보 데이터 (실제로는 DB에서 가져와야 함)
export const diseaseDatabase: Record<string, DiseaseInfo> = {
  'Apple___Apple_scab': {
    disease_name: '사과 검은점병',
    description: '사과나무의 잎과 과실에 발생하는 곰팡이병으로, 검은색 반점이 특징입니다.',
    symptoms: ['잎에 검은색 반점', '과실에 딱지 모양의 병반', '심하면 낙엽 발생'],
    treatment: ['감염된 잎과 과실 제거', '살균제 처리', '적절한 가지치기'],
    prevention: ['적절한 거리 유지', '통풍 개선', '정기적인 살균제 처리'],
    severity: 'medium'
  },
  'Apple___Black_rot': {
    disease_name: '사과 검은썩음병',
    description: '사과나무의 과실과 가지에 발생하는 심각한 곰팡이병입니다.',
    symptoms: ['과실에 검은색 썩음', '가지에 궤양', '잎에 갈색 반점'],
    treatment: ['감염된 부분 제거', '살균제 처리', '나무 전체 소독'],
    prevention: ['적절한 가지치기', '과실 솎기', '정기적인 살균제 처리'],
    severity: 'high'
  },
  'Apple___Cedar_apple_rust': {
    disease_name: '사과 삼나무 녹병',
    description: '사과나무와 삼나무 사이에서 발생하는 녹병으로, 주황색 반점이 특징입니다.',
    symptoms: ['잎에 주황색 반점', '과실에 딱지 모양 병반', '삼나무에 갈색 괴사'],
    treatment: ['감염된 잎과 과실 제거', '살균제 처리', '삼나무와의 거리 확보'],
    prevention: ['삼나무와의 거리 유지', '정기적인 살균제 처리', '적절한 가지치기'],
    severity: 'medium'
  },
  'Apple___healthy': {
    disease_name: '건강한 사과',
    description: '병충해가 없는 건강한 상태입니다.',
    symptoms: [],
    treatment: [],
    prevention: ['정기적인 관리', '적절한 수분 공급', '균형 잡힌 비료 사용'],
    severity: 'low'
  },
  'Blueberry___healthy': {
    disease_name: '건강한 블루베리',
    description: '병충해가 없는 건강한 상태입니다.',
    symptoms: [],
    treatment: [],
    prevention: ['정기적인 관리', '적절한 수분 공급', '균형 잡힌 비료 사용'],
    severity: 'low'
  },
  'Cherry___Powdery_mildew': {
    disease_name: '체리 흰가루병',
    description: '체리나무의 잎과 과실에 발생하는 곰팡이병으로, 흰색 가루 모양이 특징입니다.',
    symptoms: ['잎에 흰색 가루 모양', '과실에 흰색 반점', '잎이 뒤틀림'],
    treatment: ['감염된 잎과 과실 제거', '살균제 처리', '적절한 가지치기'],
    prevention: ['적절한 거리 유지', '통풍 개선', '정기적인 살균제 처리'],
    severity: 'medium'
  },
  'Cherry___healthy': {
    disease_name: '건강한 체리',
    description: '병충해가 없는 건강한 상태입니다.',
    symptoms: [],
    treatment: [],
    prevention: ['정기적인 관리', '적절한 수분 공급', '균형 잡힌 비료 사용'],
    severity: 'low'
  },
  'Corn___Cercospora_leaf_spot Gray_leaf_spot': {
    disease_name: '옥수수 세르코스포라 잎점무늬병',
    description: '옥수수의 잎에 발생하는 곰팡이병으로, 회색 반점이 특징입니다.',
    symptoms: ['잎에 회색 반점', '심하면 낙엽', '수량 감소'],
    treatment: ['감염된 잎 제거', '살균제 처리', '적절한 거리 유지'],
    prevention: ['적절한 거리 유지', '통풍 개선', '정기적인 살균제 처리'],
    severity: 'medium'
  },
  'Corn___Common_rust': {
    disease_name: '옥수수 일반녹병',
    description: '옥수수의 잎에 발생하는 녹병으로, 주황색 반점이 특징입니다.',
    symptoms: ['잎에 주황색 반점', '심하면 낙엽', '수량 감소'],
    treatment: ['감염된 잎 제거', '살균제 처리', '적절한 거리 유지'],
    prevention: ['적절한 거리 유지', '통풍 개선', '정기적인 살균제 처리'],
    severity: 'medium'
  },
  'Corn___Northern_Leaf_Blight': {
    disease_name: '옥수수 북부잎마름병',
    description: '옥수수의 잎에 발생하는 곰팡이병으로, 긴 갈색 반점이 특징입니다.',
    symptoms: ['잎에 긴 갈색 반점', '심하면 낙엽', '수량 감소'],
    treatment: ['감염된 잎 제거', '살균제 처리', '적절한 거리 유지'],
    prevention: ['적절한 거리 유지', '통풍 개선', '정기적인 살균제 처리'],
    severity: 'medium'
  },
  'Corn___healthy': {
    disease_name: '건강한 옥수수',
    description: '병충해가 없는 건강한 상태입니다.',
    symptoms: [],
    treatment: [],
    prevention: ['정기적인 관리', '적절한 수분 공급', '균형 잡힌 비료 사용'],
    severity: 'low'
  },
  'Cucumber___Downy_mildew': {
    disease_name: '오이 노균병',
    description: '오이의 잎에 발생하는 곰팡이병으로, 노란색 반점과 회색 곰팡이가 특징입니다.',
    symptoms: ['잎에 노란색 반점', '잎 뒷면에 회색 곰팡이', '심하면 낙엽'],
    treatment: ['감염된 잎 제거', '살균제 처리', '적절한 거리 유지'],
    prevention: ['적절한 거리 유지', '통풍 개선', '정기적인 살균제 처리'],
    severity: 'medium'
  },
  'Cucumber___Healthy': {
    disease_name: '건강한 오이',
    description: '병충해가 없는 건강한 상태입니다.',
    symptoms: [],
    treatment: [],
    prevention: ['정기적인 관리', '적절한 수분 공급', '균형 잡힌 비료 사용'],
    severity: 'low'
  },
  'Cucumber___Powdery_mildew': {
    disease_name: '오이 흰가루병',
    description: '오이의 잎에 발생하는 곰팡이병으로, 흰색 가루 모양이 특징입니다.',
    symptoms: ['잎에 흰색 가루 모양', '잎이 뒤틀림', '심하면 낙엽'],
    treatment: ['감염된 잎 제거', '살균제 처리', '적절한 거리 유지'],
    prevention: ['적절한 거리 유지', '통풍 개선', '정기적인 살균제 처리'],
    severity: 'medium'
  },
  'Grape___Black_rot': {
    disease_name: '포도 검은썩음병',
    description: '포도의 잎과 과실에 발생하는 심각한 곰팡이병입니다.',
    symptoms: ['잎에 갈색 반점', '과실에 검은색 썩음', '전체적인 시들음'],
    treatment: ['감염된 잎과 과실 제거', '살균제 처리', '적절한 가지치기'],
    prevention: ['적절한 거리 유지', '통풍 개선', '정기적인 살균제 처리'],
    severity: 'high'
  },
  'Grape___Esca_(Black_Measles)': {
    disease_name: '포도 에스카병',
    description: '포도의 줄기와 잎에 발생하는 심각한 곰팡이병입니다.',
    symptoms: ['줄기에 궤양', '잎에 노란색 반점', '전체적인 시들음'],
    treatment: ['감염된 부분 제거', '살균제 처리', '나무 전체 소독'],
    prevention: ['적절한 가지치기', '정기적인 살균제 처리', '건강한 묘목 사용'],
    severity: 'high'
  },
  'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
    disease_name: '포도 잎마름병',
    description: '포도의 잎에 발생하는 곰팡이병으로, 갈색 반점이 특징입니다.',
    symptoms: ['잎에 갈색 반점', '심하면 낙엽', '수량 감소'],
    treatment: ['감염된 잎 제거', '살균제 처리', '적절한 가지치기'],
    prevention: ['적절한 거리 유지', '통풍 개선', '정기적인 살균제 처리'],
    severity: 'medium'
  },
  'Grape___healthy': {
    disease_name: '건강한 포도',
    description: '병충해가 없는 건강한 상태입니다.',
    symptoms: [],
    treatment: [],
    prevention: ['정기적인 관리', '적절한 수분 공급', '균형 잡힌 비료 사용'],
    severity: 'low'
  },
  'Orange___Haunglongbing_(Citrus_greening)': {
    disease_name: '오렌지 황룡병',
    description: '오렌지나무에 발생하는 심각한 세균병으로, 잎이 노랗게 변합니다.',
    symptoms: ['잎이 노랗게 변함', '과실이 작아짐', '전체적인 시들음'],
    treatment: ['감염된 나무 제거', '살균제 처리', '나무 전체 소독'],
    prevention: ['건강한 묘목 사용', '정기적인 살균제 처리', '적절한 관리'],
    severity: 'high'
  },
  'Peach___Bacterial_spot': {
    disease_name: '복숭아 세균점무늬병',
    description: '복숭아나무의 잎과 과실에 발생하는 세균병입니다.',
    symptoms: ['잎에 갈색 반점', '과실에 딱지 모양 병반', '심하면 낙엽'],
    treatment: ['감염된 잎과 과실 제거', '살균제 처리', '적절한 가지치기'],
    prevention: ['적절한 거리 유지', '통풍 개선', '정기적인 살균제 처리'],
    severity: 'medium'
  },
  'Peach___healthy': {
    disease_name: '건강한 복숭아',
    description: '병충해가 없는 건강한 상태입니다.',
    symptoms: [],
    treatment: [],
    prevention: ['정기적인 관리', '적절한 수분 공급', '균형 잡힌 비료 사용'],
    severity: 'low'
  },
  'Pepper,_bell___Bacterial_spot': {
    disease_name: '피망 세균점무늬병',
    description: '피망의 잎과 과실에 발생하는 세균병입니다.',
    symptoms: ['잎에 갈색 반점', '과실에 딱지 모양 병반', '심하면 낙엽'],
    treatment: ['감염된 잎과 과실 제거', '살균제 처리', '적절한 거리 유지'],
    prevention: ['적절한 거리 유지', '통풍 개선', '정기적인 살균제 처리'],
    severity: 'medium'
  },
  'Pepper,_bell___healthy': {
    disease_name: '건강한 피망',
    description: '병충해가 없는 건강한 상태입니다.',
    symptoms: [],
    treatment: [],
    prevention: ['정기적인 관리', '적절한 수분 공급', '균형 잡힌 비료 사용'],
    severity: 'low'
  },
  'Potato___Early_blight': {
    disease_name: '감자 겹무늬병',
    description: '감자의 잎에 발생하는 곰팡이병으로, 동심원 모양의 반점이 특징입니다.',
    symptoms: ['잎에 동심원 모양 반점', '심하면 낙엽', '수량 감소'],
    treatment: ['감염된 잎 제거', '살균제 처리', '적절한 거리 유지'],
    prevention: ['적절한 거리 유지', '통풍 개선', '정기적인 살균제 처리'],
    severity: 'medium'
  },
  'Potato___Late_blight': {
    disease_name: '감자 역병',
    description: '감자의 잎과 줄기에 발생하는 심각한 곰팡이병입니다.',
    symptoms: ['잎에 물에 젖은 반점', '줄기에 갈색 반점', '전체적인 시들음'],
    treatment: ['감염된 식물 제거', '살균제 처리', '토양 소독'],
    prevention: ['적절한 거리 유지', '통풍 개선', '정기적인 살균제 처리'],
    severity: 'high'
  },
  'Potato___healthy': {
    disease_name: '건강한 감자',
    description: '병충해가 없는 건강한 상태입니다.',
    symptoms: [],
    treatment: [],
    prevention: ['정기적인 관리', '적절한 수분 공급', '균형 잡힌 비료 사용'],
    severity: 'low'
  },
  'Raspberry___healthy': {
    disease_name: '건강한 라즈베리',
    description: '병충해가 없는 건강한 상태입니다.',
    symptoms: [],
    treatment: [],
    prevention: ['정기적인 관리', '적절한 수분 공급', '균형 잡힌 비료 사용'],
    severity: 'low'
  },
  'Rice___Bacterial_Leaf_Blight': {
    disease_name: '벼 세균성 잎마름병',
    description: '벼의 잎에 발생하는 심각한 세균병입니다.',
    symptoms: ['잎에 노란색 반점', '심하면 잎이 마름', '수량 감소'],
    treatment: ['감염된 잎 제거', '살균제 처리', '적절한 거리 유지'],
    prevention: ['적절한 거리 유지', '통풍 개선', '정기적인 살균제 처리'],
    severity: 'high'
  },
  'Rice___Brown_Spot': {
    disease_name: '벼 갈색점무늬병',
    description: '벼의 잎에 발생하는 곰팡이병으로, 갈색 반점이 특징입니다.',
    symptoms: ['잎에 갈색 반점', '심하면 낙엽', '수량 감소'],
    treatment: ['감염된 잎 제거', '살균제 처리', '적절한 거리 유지'],
    prevention: ['적절한 거리 유지', '통풍 개선', '정기적인 살균제 처리'],
    severity: 'medium'
  },
  'Rice___Healthy': {
    disease_name: '건강한 벼',
    description: '병충해가 없는 건강한 상태입니다.',
    symptoms: [],
    treatment: [],
    prevention: ['정기적인 관리', '적절한 수분 공급', '균형 잡힌 비료 사용'],
    severity: 'low'
  },
  'Rice___Leaf_Blast': {
    disease_name: '벼 도열병',
    description: '벼의 잎에 발생하는 심각한 곰팡이병입니다.',
    symptoms: ['잎에 다이아몬드 모양 반점', '심하면 잎이 마름', '수량 감소'],
    treatment: ['감염된 잎 제거', '살균제 처리', '적절한 거리 유지'],
    prevention: ['적절한 거리 유지', '통풍 개선', '정기적인 살균제 처리'],
    severity: 'high'
  },
  'Rice___Leaf_Scald': {
    disease_name: '벼 잎마름병',
    description: '벼의 잎에 발생하는 곰팡이병으로, 잎 가장자리가 마름니다.',
    symptoms: ['잎 가장자리 마름', '심하면 잎이 마름', '수량 감소'],
    treatment: ['감염된 잎 제거', '살균제 처리', '적절한 거리 유지'],
    prevention: ['적절한 거리 유지', '통풍 개선', '정기적인 살균제 처리'],
    severity: 'medium'
  },
  'Rice___Sheath_Blight': {
    disease_name: '벼 잎집무늬마름병',
    description: '벼의 잎집에 발생하는 곰팡이병입니다.',
    symptoms: ['잎집에 갈색 반점', '심하면 잎이 마름', '수량 감소'],
    treatment: ['감염된 잎 제거', '살균제 처리', '적절한 거리 유지'],
    prevention: ['적절한 거리 유지', '통풍 개선', '정기적인 살균제 처리'],
    severity: 'medium'
  },
  'Soybean___healthy': {
    disease_name: '건강한 콩',
    description: '병충해가 없는 건강한 상태입니다.',
    symptoms: [],
    treatment: [],
    prevention: ['정기적인 관리', '적절한 수분 공급', '균형 잡힌 비료 사용'],
    severity: 'low'
  },
  'Squash___Powdery_mildew': {
    disease_name: '호박 흰가루병',
    description: '호박의 잎에 발생하는 곰팡이병으로, 흰색 가루 모양이 특징입니다.',
    symptoms: ['잎에 흰색 가루 모양', '잎이 뒤틀림', '심하면 낙엽'],
    treatment: ['감염된 잎 제거', '살균제 처리', '적절한 거리 유지'],
    prevention: ['적절한 거리 유지', '통풍 개선', '정기적인 살균제 처리'],
    severity: 'medium'
  },
  'Strawberry___Leaf_scorch': {
    disease_name: '딸기 잎마름병',
    description: '딸기의 잎에 발생하는 곰팡이병으로, 잎 가장자리가 마름니다.',
    symptoms: ['잎 가장자리 마름', '심하면 잎이 마름', '수량 감소'],
    treatment: ['감염된 잎 제거', '살균제 처리', '적절한 거리 유지'],
    prevention: ['적절한 거리 유지', '통풍 개선', '정기적인 살균제 처리'],
    severity: 'medium'
  },
  'Strawberry___healthy': {
    disease_name: '건강한 딸기',
    description: '병충해가 없는 건강한 상태입니다.',
    symptoms: [],
    treatment: [],
    prevention: ['정기적인 관리', '적절한 수분 공급', '균형 잡힌 비료 사용'],
    severity: 'low'
  },
  'Tomato___Bacterial_spot': {
    disease_name: '토마토 세균점무늬병',
    description: '토마토의 잎과 과실에 발생하는 세균병입니다.',
    symptoms: ['잎에 갈색 반점', '과실에 딱지 모양 병반', '심하면 낙엽'],
    treatment: ['감염된 잎과 과실 제거', '살균제 처리', '적절한 거리 유지'],
    prevention: ['적절한 거리 유지', '통풍 개선', '정기적인 살균제 처리'],
    severity: 'medium'
  },
  'Tomato___Early_blight': {
    disease_name: '토마토 겹무늬병',
    description: '토마토의 잎과 줄기에 발생하는 곰팡이병으로, 동심원 모양의 반점이 특징입니다.',
    symptoms: ['잎에 동심원 모양 반점', '줄기에 갈색 반점', '심하면 낙엽'],
    treatment: ['감염된 잎 제거', '살균제 처리', '적절한 거리 유지'],
    prevention: ['적절한 거리 유지', '통풍 개선', '정기적인 살균제 처리'],
    severity: 'medium'
  },
  'Tomato___Late_blight': {
    disease_name: '토마토 역병',
    description: '토마토의 잎과 과실에 발생하는 심각한 곰팡이병입니다.',
    symptoms: ['잎에 물에 젖은 반점', '과실에 갈색 썩음', '전체적인 시들음'],
    treatment: ['감염된 식물 제거', '살균제 처리', '토양 소독'],
    prevention: ['적절한 거리 유지', '통풍 개선', '정기적인 살균제 처리'],
    severity: 'high'
  },
  'Tomato___Leaf_Mold': {
    disease_name: '토마토 잎곰팡이병',
    description: '토마토의 잎에 발생하는 곰팡이병으로, 노란색 반점과 회색 곰팡이가 특징입니다.',
    symptoms: ['잎에 노란색 반점', '잎 뒷면에 회색 곰팡이', '심하면 낙엽'],
    treatment: ['감염된 잎 제거', '살균제 처리', '적절한 거리 유지'],
    prevention: ['적절한 거리 유지', '통풍 개선', '정기적인 살균제 처리'],
    severity: 'medium'
  },
  'Tomato___Septoria_leaf_spot': {
    disease_name: '토마토 세프토리아 잎점무늬병',
    description: '토마토의 잎에 발생하는 곰팡이병으로, 작은 갈색 반점이 특징입니다.',
    symptoms: ['잎에 작은 갈색 반점', '심하면 낙엽', '수량 감소'],
    treatment: ['감염된 잎 제거', '살균제 처리', '적절한 거리 유지'],
    prevention: ['적절한 거리 유지', '통풍 개선', '정기적인 살균제 처리'],
    severity: 'medium'
  },
  'Tomato___Spider_mites Two-spotted_spider_mite': {
    disease_name: '토마토 응애',
    description: '토마토에 발생하는 해충으로, 잎을 갉아먹어 피해를 줍니다.',
    symptoms: ['잎에 작은 반점', '잎이 노랗게 변함', '거미줄 모양'],
    treatment: ['살충제 처리', '감염된 잎 제거', '적절한 거리 유지'],
    prevention: ['적절한 거리 유지', '정기적인 살충제 처리', '건강한 묘목 사용'],
    severity: 'medium'
  },
  'Tomato___Target_Spot': {
    disease_name: '토마토 표적점무늬병',
    description: '토마토의 잎에 발생하는 곰팡이병으로, 표적 모양의 반점이 특징입니다.',
    symptoms: ['잎에 표적 모양 반점', '심하면 낙엽', '수량 감소'],
    treatment: ['감염된 잎 제거', '살균제 처리', '적절한 거리 유지'],
    prevention: ['적절한 거리 유지', '통풍 개선', '정기적인 살균제 처리'],
    severity: 'medium'
  },
  'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
    disease_name: '토마토 황화잎말림바이러스',
    description: '토마토에 발생하는 바이러스병으로, 잎이 노랗게 변하고 말립니다.',
    symptoms: ['잎이 노랗게 변함', '잎이 말림', '전체적인 시들음'],
    treatment: ['감염된 식물 제거', '바이러스 치료제 처리', '토양 소독'],
    prevention: ['건강한 묘목 사용', '정기적인 살충제 처리', '적절한 관리'],
    severity: 'high'
  },
  'Tomato___Tomato_mosaic_virus': {
    disease_name: '토마토 모자이크바이러스',
    description: '토마토에 발생하는 바이러스병으로, 잎에 모자이크 패턴이 나타납니다.',
    symptoms: ['잎에 모자이크 패턴', '잎이 뒤틀림', '전체적인 시들음'],
    treatment: ['감염된 식물 제거', '바이러스 치료제 처리', '토양 소독'],
    prevention: ['건강한 묘목 사용', '정기적인 살충제 처리', '적절한 관리'],
    severity: 'high'
  },
  'Tomato___healthy': {
    disease_name: '건강한 토마토',
    description: '병충해가 없는 건강한 상태입니다.',
    symptoms: [],
    treatment: [],
    prevention: ['정기적인 관리', '적절한 수분 공급', '균형 잡힌 비료 사용'],
    severity: 'low'
  }
}

// 병충해 정보 조회 함수
export function getDiseaseInfo(diseaseLabel: string): DiseaseInfo | null {
  return diseaseDatabase[diseaseLabel] || null
}

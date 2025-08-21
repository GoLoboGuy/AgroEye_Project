<template>
  <div class="analysis-result">
    <div v-if="loading" class="loading-container">
      <div class="loading-spinner"></div>
      <p>이미지를 분석하고 있습니다...</p>
    </div>

    <div v-else-if="error" class="error-container">
      <div class="error-icon">⚠️</div>
      <h3>분석 중 오류가 발생했습니다</h3>
      <p>{{ error }}</p>
      <button @click="$emit('retry')" class="retry-btn">다시 시도</button>
    </div>

    <div v-else-if="result" class="result-container">
      <!-- 분석 결과 요약 -->
      <div class="result-summary">
        <div class="result-header">
          <h3>분석 결과</h3>
          <div class="confidence-badge" :class="getConfidenceClass(result.picked.confidence)">
            {{ Math.round(result.picked.confidence * 100) }}%
          </div>
        </div>

        <div class="prediction-info">
          <div class="prediction-main">
            <h4>{{ getDiseaseName(result.picked.label) }}</h4>
            <p class="model-used">사용된 모델: {{ result.picked.model }}</p>
          </div>

          <div class="model-comparison">
            <div class="model-result">
              <span class="model-name">MobileNetV2:</span>
              <span class="model-prediction">{{ getDiseaseName(result.mobilenet.label) }}</span>
              <span class="model-confidence">{{ Math.round(result.mobilenet.confidence * 100) }}%</span>
            </div>
            <div class="model-result">
              <span class="model-name">ResNet50:</span>
              <span class="model-prediction">{{ getDiseaseName(result.resnet50.label) }}</span>
              <span class="model-confidence">{{ Math.round(result.resnet50.confidence * 100) }}%</span>
            </div>
          </div>
        </div>
      </div>

      <!-- 병충해 상세 정보 -->
      <div v-if="diseaseInfo" class="disease-details">
        <div class="disease-header">
          <h3>병충해 정보</h3>
          <div class="severity-badge" :class="diseaseInfo.severity">
            {{ getSeverityText(diseaseInfo.severity) }}
          </div>
        </div>

        <div class="disease-content">
          <div class="disease-description">
            <h4>설명</h4>
            <p>{{ diseaseInfo.description }}</p>
          </div>

          <div v-if="diseaseInfo.symptoms.length > 0" class="disease-section">
            <h4>증상</h4>
            <ul class="symptom-list">
              <li v-for="symptom in diseaseInfo.symptoms" :key="symptom">
                {{ symptom }}
              </li>
            </ul>
          </div>

          <div v-if="diseaseInfo.treatment.length > 0" class="disease-section">
            <h4>치료 방법</h4>
            <ul class="treatment-list">
              <li v-for="treatment in diseaseInfo.treatment" :key="treatment">
                {{ treatment }}
              </li>
            </ul>
          </div>

          <div v-if="diseaseInfo.prevention.length > 0" class="disease-section">
            <h4>예방 방법</h4>
            <ul class="prevention-list">
              <li v-for="prevention in diseaseInfo.prevention" :key="prevention">
                {{ prevention }}
              </li>
            </ul>
          </div>
        </div>
      </div>

      <!-- 건강한 상태인 경우 -->
      <div v-else class="healthy-status">
        <div class="healthy-icon">🌱</div>
        <h3>건강한 상태입니다!</h3>
        <p>{{ getDiseaseName(result.picked.label) }}</p>
        <div class="health-tips">
          <h4>건강 유지를 위한 팁</h4>
          <ul>
            <li>정기적인 관찰과 관리</li>
            <li>적절한 수분 공급</li>
            <li>균형 잡힌 비료 사용</li>
            <li>적절한 햇빛과 통풍</li>
          </ul>
        </div>
      </div>

      <!-- 액션 버튼들 -->
      <div class="action-buttons">
        <button @click="$emit('new-analysis')" class="new-analysis-btn">
          새로운 분석
        </button>
        <button @click="$emit('save-result')" class="save-result-btn">
          결과 저장
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { getDiseaseInfo } from '../services/api'
import type { ModelPrediction } from '../services/api'

interface Props {
  result: ModelPrediction | null
  loading: boolean
  error: string | null
}

const props = defineProps<Props>()
defineEmits<{
  retry: []
  'new-analysis': []
  'save-result': []
}>()

// 병충해 정보 계산
const diseaseInfo = computed(() => {
  if (!props.result) return null
  return getDiseaseInfo(props.result.picked.label)
})

// 병충해 이름 변환
const getDiseaseName = (label: string): string => {
  const diseaseInfo = getDiseaseInfo(label)
  return diseaseInfo ? diseaseInfo.disease_name : label
}

// 신뢰도에 따른 클래스 반환
const getConfidenceClass = (confidence: number): string => {
  if (confidence >= 0.8) return 'high'
  if (confidence >= 0.6) return 'medium'
  return 'low'
}



// 심각도 텍스트 반환
const getSeverityText = (severity: string): string => {
  switch (severity) {
    case 'low': return '경미'
    case 'medium': return '보통'
    case 'high': return '심각'
    default: return '알 수 없음'
  }
}
</script>

<style scoped>
.analysis-result {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
}

.loading-container {
  text-align: center;
  padding: 60px 20px;
}

.loading-spinner {
  width: 50px;
  height: 50px;
  border: 4px solid #f3f3f3;
  border-top: 4px solid #3498db;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto 20px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.error-container {
  text-align: center;
  padding: 40px 20px;
  background-color: rgba(254, 242, 242, 0.98);
  border: 1px solid rgba(254, 202, 202, 0.8);
  border-radius: 8px;
  backdrop-filter: blur(5px);
}

.error-icon {
  font-size: 48px;
  margin-bottom: 16px;
}

.retry-btn {
  background-color: var(--color-danger);
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 6px;
  cursor: pointer;
  margin-top: 16px;
}

.retry-btn:hover {
  background-color: #b91c1c;
}

.result-container {
  background-color: rgba(255, 255, 255, 0.98);
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  overflow: hidden;
  backdrop-filter: blur(5px);
}

.result-summary {
  padding: 24px;
  border-bottom: 1px solid #e5e7eb;
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.result-header h3 {
  margin: 0;
  color: #1f2937;
  font-size: 20px;
}

.confidence-badge {
  padding: 6px 12px;
  border-radius: 20px;
  font-weight: 600;
  font-size: 14px;
}

.confidence-badge.high {
  background-color: #dcfce7;
  color: #166534;
}

.confidence-badge.medium {
  background-color: #fef3c7;
  color: #92400e;
}

.confidence-badge.low {
  background-color: #fee2e2;
  color: #991b1b;
}

.prediction-info {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.prediction-main h4 {
  margin: 0 0 8px 0;
  color: #1f2937;
  font-size: 18px;
}

.model-used {
  color: #6b7280;
  margin: 0;
}

.model-comparison {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 16px;
  background-color: #f9fafb;
  border-radius: 8px;
}

.model-result {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 4px 0;
}

.model-name {
  font-weight: 600;
  color: #374151;
  min-width: 100px;
}

.model-prediction {
  flex: 1;
  color: #1f2937;
}

.model-confidence {
  font-weight: 600;
  color: #059669;
  min-width: 50px;
  text-align: right;
}

.disease-details {
  padding: 24px;
}

.disease-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.disease-header h3 {
  margin: 0;
  color: #1f2937;
  font-size: 20px;
}

.severity-badge {
  padding: 6px 12px;
  border-radius: 20px;
  font-weight: 600;
  font-size: 14px;
}

.severity-badge.low {
  background-color: #dcfce7;
  color: #166534;
}

.severity-badge.medium {
  background-color: #fef3c7;
  color: #92400e;
}

.severity-badge.high {
  background-color: #fee2e2;
  color: #991b1b;
}

.disease-content {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.disease-description h4,
.disease-section h4 {
  margin: 0 0 12px 0;
  color: #1f2937;
  font-size: 16px;
}

.disease-description p {
  margin: 0;
  color: #4b5563;
  line-height: 1.6;
}

.symptom-list,
.treatment-list,
.prevention-list {
  margin: 0;
  padding-left: 20px;
  color: #4b5563;
}

.symptom-list li,
.treatment-list li,
.prevention-list li {
  margin-bottom: 8px;
  line-height: 1.5;
}

.healthy-status {
  text-align: center;
  padding: 40px 24px;
}

.healthy-icon {
  font-size: 64px;
  margin-bottom: 16px;
}

.healthy-status h3 {
  margin: 0 0 8px 0;
  color: #059669;
  font-size: 24px;
}

.healthy-status p {
  margin: 0 0 24px 0;
  color: #6b7280;
}

.health-tips {
  text-align: left;
  background-color: rgba(240, 253, 244, 0.98);
  padding: 20px;
  border-radius: 8px;
  border: 1px solid rgba(187, 247, 208, 0.8);
  backdrop-filter: blur(5px);
}

.health-tips h4 {
  margin: 0 0 16px 0;
  color: #166534;
}

.health-tips ul {
  margin: 0;
  padding-left: 20px;
  color: #166534;
}

.health-tips li {
  margin-bottom: 8px;
}

.action-buttons {
  display: flex;
  gap: 12px;
  padding: 24px;
  border-top: 1px solid #e5e7eb;
  justify-content: center;
}

.new-analysis-btn,
.save-result-btn {
  padding: 12px 24px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 500;
  transition: background-color 0.2s;
}

.new-analysis-btn {
  background-color: var(--color-primary);
  color: white;
}

.new-analysis-btn:hover {
  background-color: #2563eb;
}

.save-result-btn {
  background-color: var(--color-success);
  color: white;
}

.save-result-btn:hover {
  background-color: #059669;
}

@media (max-width: 768px) {
  .analysis-result {
    padding: 16px;
  }

  .result-header,
  .disease-header {
    flex-direction: column;
    gap: 12px;
    align-items: flex-start;
  }

  .action-buttons {
    flex-direction: column;
  }

  .model-comparison {
    padding: 12px;
  }

  .model-result {
    flex-direction: column;
    align-items: flex-start;
    gap: 4px;
  }

  .model-confidence {
    text-align: left;
  }
}
</style>


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
          <div class="confidence-badge" :class="getConfidenceClass(result.confidence)">
            {{ Math.round(result.confidence * 100) }}%
          </div>
        </div>

        <div class="prediction-info">
          <div class="prediction-main">
            <!-- 건강한 상태인 경우 이모지와 함께 표시 -->
            <div v-if="result.class_name.includes('healthy')" class="healthy-status-inline">
              <span class="healthy-icon-inline">🌱</span>
              <h2 class="healthy-status-text">건강한 상태입니다!</h2>
            </div>
            <!-- Unknown 상태인 경우 느낌표 표지판 이모지와 함께 표시 -->
            <div v-else-if="result.class_name === 'Unknown'" class="unknown-status-inline">
              <span class="unknown-icon-inline">🚫</span>
              <h2 class="unknown-status-text">분석을 할 수 없습니다</h2>
            </div>
            <!-- 질병인 경우 경고 이모지와 함께 표시 -->
            <div v-else class="disease-status-inline">
              <span class="disease-icon-inline">⚠️</span>
              <h2 class="disease-status-text">질병이 감지되었습니다!</h2>
              <h4 class="disease-name">{{ getDiseaseName(result.class_name) }}</h4>
            </div>
            <p class="model-used">분석 완료</p>
          </div>
        </div>
      </div>

      <!-- 추천사항 -->
      <div v-if="result.recomm" class="recommendation-section">
        <div class="recommendation-header">
          <h3>추천사항</h3>
        </div>
        <div class="recommendation-content">
          <p>{{ result.recomm }}</p>
        </div>
      </div>

      <!-- 소스 정보 -->
      <div v-if="result.sources && result.sources.length > 0" class="sources-section">
        <div class="sources-header">
          <h3>참고 자료</h3>
        </div>
        <div class="sources-content">
          <!-- 첫 번째 카드 (항상 표시) -->
          <div v-if="result.sources[0]" class="source-item">
            <div class="source-title">{{ result.sources[0].title || '참고 자료' }}</div>
            <div class="source-snippet">{{ result.sources[0].snippet }}</div>
            <div class="source-meta">
              <span v-if="result.sources[0].page">페이지: {{ result.sources[0].page }}</span>
            </div>
          </div>

          <!-- 더 보기/접기 버튼 -->
          <div v-if="result.sources.length > 1" class="toggle-sources">
            <button @click="showAllSources = !showAllSources" class="toggle-btn">
              <span v-if="!showAllSources">
                더 보기 ({{ result.sources.length - 1 }}개 더)
              </span>
              <span v-else>
                접기
              </span>
              <span class="toggle-icon" :class="{ 'expanded': showAllSources }">▼</span>
            </button>
          </div>

          <!-- 나머지 카드들 (접을 수 있음) -->
          <div v-if="showAllSources" class="additional-sources">
            <div v-for="(source, index) in result.sources.slice(1)" :key="index + 1" class="source-item">
              <div class="source-title">{{ source.title || '참고 자료' }}</div>
              <div class="source-snippet">{{ source.snippet }}</div>
              <div class="source-meta">
                <span v-if="source.page">페이지: {{ source.page }}</span>
              </div>
            </div>
          </div>
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
import { ref } from 'vue'
import type { PredictResponse } from '../services/api'

interface Props {
  result: PredictResponse | null
  loading: boolean
  error: string | null
}

defineProps<Props>()
defineEmits<{
  retry: []
  'new-analysis': []
  'save-result': []
}>()

// 참고자료 펼치기/접기 상태
const showAllSources = ref(false)

// 질병 이름 변환
const getDiseaseName = (className: string): string => {
  return className || '알 수 없는 질병'
}

// 신뢰도에 따른 클래스 반환
const getConfidenceClass = (confidence: number): string => {
  if (confidence >= 0.8) return 'high'
  if (confidence >= 0.6) return 'medium'
  return 'low'
}
</script>

<style scoped>
.analysis-result {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
  color: #000000;
  animation: fadeInUp 0.6s ease-out;
}

/* 공통 애니메이션 */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

/* 공통 컨테이너 스타일 */
.card-container {
  background: rgba(255, 255, 255, 0.95);
  border-radius: 16px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1), 0 2px 8px rgba(0, 0, 0, 0.05);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  position: relative;
  overflow: hidden;
  transition: all 0.3s ease;
}

.card-container::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
  border-radius: 16px 16px 0 0;
}

.card-container:hover {
  transform: translateY(-2px);
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15), 0 4px 12px rgba(0, 0, 0, 0.1);
}

.loading-container {
  @apply card-container;
  text-align: center;
  padding: 60px 20px;
  color: #000000;
}

.loading-spinner {
  width: 50px;
  height: 50px;
  border: 4px solid #f3f3f3;
  border-top: 4px solid #667eea;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto 20px;
}

.error-container {
  @apply card-container;
  text-align: center;
  padding: 40px 20px;
  color: #000000;
}

.error-container::before {
  background: linear-gradient(90deg, #dc3545, #e74c3c, #f093fb);
}

.error-icon {
  font-size: 48px;
  margin-bottom: 16px;
}

.retry-btn {
  background: linear-gradient(135deg, #dc3545, #e74c3c);
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 8px;
  cursor: pointer;
  margin-top: 16px;
  font-weight: 600;
  transition: all 0.2s ease;
}

.retry-btn:hover {
  transform: translateY(-1px);
  opacity: 0.9;
}

.result-container {
  @apply card-container;
}

.result-summary {
  padding: 24px;
  border-bottom: 1px solid rgba(102, 126, 234, 0.1);
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.result-header h3 {
  margin: 0;
  color: #2d3748;
  font-size: 20px;
  font-weight: 600;
}

.confidence-badge {
  padding: 8px 16px;
  border-radius: 20px;
  font-weight: 600;
  font-size: 14px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.confidence-badge.high {
  background: linear-gradient(135deg, #dcfce7, #bbf7d0);
  color: #166534;
}

.confidence-badge.medium {
  background: linear-gradient(135deg, #fef3c7, #fde68a);
  color: #92400e;
}

.confidence-badge.low {
  background: linear-gradient(135deg, #fee2e2, #fecaca);
  color: #991b1b;
}

.prediction-info {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.prediction-main h4 {
  margin: 0 0 8px 0;
  color: #2d3748;
  font-size: 18px;
  font-weight: 600;
}

.healthy-status-inline {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  gap: 16px;
  margin: 20px 0;
  padding: 20px;
}

.healthy-icon-inline {
  font-size: 48px;
}

.healthy-status-text {
  margin: 0;
  color: #059669;
  font-size: 28px;
  font-weight: 700;
  text-align: center;
}

.disease-status-inline {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  gap: 16px;
  margin: 20px 0;
  padding: 20px;
}

.disease-icon-inline {
  font-size: 48px;
}

.disease-status-text {
  margin: 0;
  color: #d97706;
  font-size: 28px;
  font-weight: 700;
  text-align: center;
}

.disease-name {
  margin: 0;
  color: #2d3748;
  font-size: 18px;
  font-weight: 600;
  text-align: center;
}

.unknown-status-inline {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  gap: 16px;
  margin: 20px 0;
  padding: 20px;
}

.unknown-icon-inline {
  font-size: 48px;
}

.unknown-status-text {
  margin: 0;
  color: #dc2626;
  font-size: 28px;
  font-weight: 700;
  text-align: center;
}

.model-used {
  color: #6b7280;
  margin: 0;
}

.recommendation-section {
  padding: 24px;
  border-top: 1px solid rgba(102, 126, 234, 0.1);
}

.recommendation-header h3 {
  margin: 0 0 16px 0;
  color: #2d3748;
  font-size: 18px;
  font-weight: 600;
}

.recommendation-content p {
  margin: 0;
  line-height: 1.6;
  color: #4a5568;
  white-space: pre-line;
}

.sources-section {
  padding: 24px;
  border-top: 1px solid rgba(102, 126, 234, 0.1);
}

.sources-header h3 {
  margin: 0 0 16px 0;
  color: #2d3748;
  font-size: 18px;
  font-weight: 600;
}

.sources-content {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.source-item {
  padding: 16px;
  background: rgba(255, 255, 255, 0.8);
  border-radius: 12px;
  border-left: 4px solid #667eea;
  transition: all 0.3s ease;
  border: 1px solid rgba(102, 126, 234, 0.1);
}

.source-item:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  background: rgba(255, 255, 255, 0.95);
}

.source-title {
  font-weight: 600;
  color: #2d3748;
  margin-bottom: 8px;
}

.source-snippet {
  color: #4a5568;
  line-height: 1.5;
  margin-bottom: 8px;
  font-size: 14px;
}

.source-meta {
  display: flex;
  gap: 16px;
  font-size: 12px;
  color: #6b7280;
}

.toggle-sources {
  display: flex;
  justify-content: center;
  margin: 16px 0;
}

.toggle-btn {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 600;
  font-size: 14px;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  gap: 8px;
}

.toggle-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}

.toggle-icon {
  transition: transform 0.3s ease;
  font-size: 12px;
}

.toggle-icon.expanded {
  transform: rotate(180deg);
}

.additional-sources {
  animation: slideDown 0.3s ease-out;
}

@keyframes slideDown {
  from {
    opacity: 0;
    transform: translateY(-10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
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
  color: #2d3748;
  font-size: 24px;
  font-weight: 600;
}

.healthy-status p {
  margin: 0;
  color: #4a5568;
}

.action-buttons {
  display: flex;
  gap: 12px;
  padding: 24px;
  border-top: 1px solid rgba(102, 126, 234, 0.1);
  justify-content: center;
}

/* 버튼 스타일 */
.action-buttons button {
  padding: 12px 24px;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 600;
  transition: all 0.2s ease;
  color: white;
}

.new-analysis-btn {
  background: linear-gradient(135deg, #667eea, #764ba2);
}

.save-result-btn {
  background: linear-gradient(135deg, #28a745, #20c997);
}

.action-buttons button:hover {
  transform: translateY(-1px);
  opacity: 0.9;
}

@media (max-width: 768px) {
  .analysis-result {
    max-width: 100%;
    width: 100%;
    padding: 16px;
  }

  .result-container,
  .recommendation-section,
  .sources-section,
  .action-buttons {
    border-radius: 8px;
  }

  .action-buttons {
    flex-direction: column;
  }

  .source-meta {
    flex-direction: column;
    gap: 8px;
  }
}
</style>


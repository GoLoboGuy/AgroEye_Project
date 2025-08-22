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
            <h4>{{ getDiseaseName(result.class_name) }}</h4>
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
          <div v-for="(source, index) in result.sources" :key="index" class="source-item">
            <div class="source-title">{{ source.title || '참고 자료' }}</div>
            <div class="source-snippet">{{ source.snippet }}</div>
            <div class="source-meta">
              <span v-if="source.page">페이지: {{ source.page }}</span>
              <span v-if="source.score">신뢰도: {{ Math.round(source.score * 100) }}%</span>
            </div>
          </div>
        </div>
      </div>

      <!-- 건강한 상태인 경우 -->
      <div v-if="result.class_name.includes('healthy')" class="healthy-status">
        <div class="healthy-icon">🌱</div>
        <h3>건강한 상태입니다!</h3>
        <p>{{ getDiseaseName(result.class_name) }}</p>
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

// 질병 이름 변환 (Backend에서 받은 class_name 사용)
const getDiseaseName = (className: string): string => {
  // Backend에서 받은 class_name을 그대로 사용하거나 한글로 변환
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

/* 페이드인 애니메이션 */
@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.loading-container {
  text-align: center;
  padding: 60px 20px;
  color: #000000;
  background: rgba(255, 255, 255, 0.95);
  border-radius: 16px;
  box-shadow:
    0 8px 32px rgba(0, 0, 0, 0.1),
    0 2px 8px rgba(0, 0, 0, 0.05);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  position: relative;
  overflow: hidden;
}

.loading-container::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
  border-radius: 16px 16px 0 0;
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

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.error-container {
  text-align: center;
  padding: 40px 20px;
  background: rgba(255, 255, 255, 0.95);
  border-radius: 16px;
  box-shadow:
    0 8px 32px rgba(0, 0, 0, 0.1),
    0 2px 8px rgba(0, 0, 0, 0.05);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  color: #000000;
  position: relative;
  overflow: hidden;
  transition: all 0.3s ease;
}

.error-container:hover {
  transform: translateY(-4px);
  box-shadow:
    0 12px 40px rgba(0, 0, 0, 0.15),
    0 4px 12px rgba(0, 0, 0, 0.1);
}

.error-container::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: linear-gradient(90deg, #dc3545, #e74c3c, #f093fb);
  border-radius: 16px 16px 0 0;
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
  transition: all 0.3s ease;
  box-shadow: 0 4px 15px rgba(220, 53, 69, 0.4);
}

.retry-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(220, 53, 69, 0.6);
}

.result-container {
  background: rgba(255, 255, 255, 0.95);
  border-radius: 16px;
  box-shadow:
    0 8px 32px rgba(0, 0, 0, 0.1),
    0 2px 8px rgba(0, 0, 0, 0.05);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  overflow: hidden;
  position: relative;
  transition: all 0.3s ease;
}

.result-container:hover {
  transform: translateY(-4px);
  box-shadow:
    0 12px 40px rgba(0, 0, 0, 0.15),
    0 4px 12px rgba(0, 0, 0, 0.1);
}

.result-container::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
  border-radius: 16px 16px 0 0;
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
  margin: 0 0 24px 0;
  color: #4a5568;
}

.health-tips {
  text-align: left;
  background: rgba(240, 253, 244, 0.95);
  padding: 20px;
  border-radius: 12px;
  border: 1px solid rgba(102, 126, 234, 0.1);
  backdrop-filter: blur(10px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
}

.health-tips h4 {
  margin: 0 0 16px 0;
  color: #2d3748;
  font-weight: 600;
}

.health-tips ul {
  margin: 0;
  padding-left: 20px;
  color: #4a5568;
}

.health-tips li {
  margin-bottom: 8px;
}

.action-buttons {
  display: flex;
  gap: 12px;
  padding: 24px;
  border-top: 1px solid rgba(102, 126, 234, 0.1);
  justify-content: center;
}

.new-analysis-btn,
.save-result-btn {
  padding: 12px 24px;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 600;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.new-analysis-btn {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}

.new-analysis-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
}

.save-result-btn {
  background: linear-gradient(135deg, #28a745, #20c997);
  color: white;
  box-shadow: 0 4px 15px rgba(40, 167, 69, 0.4);
}

.save-result-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(40, 167, 69, 0.6);
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


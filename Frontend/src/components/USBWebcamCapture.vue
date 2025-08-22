<template>
  <div class="usb-webcam-container">
    <h2>🌱 농작물 질병 진단 시스템</h2>
    <p class="subtitle">카메라로 작물을 촬영하여 AI가 질병을 진단해드립니다</p>

    <!-- 카메라 선택 섹션 -->
    <div class="device-selection">
      <div class="section-header">
        <div class="icon-wrapper">🎥</div>
        <h3>카메라 장치 선택</h3>
      </div>
      <div class="device-list">
        <select
          v-model="selectedDeviceId"
          @change="onDeviceChange"
          class="device-select"
          :disabled="isStreamActive"
        >
          <option value="">카메라 장치를 선택하세요</option>
          <option
            v-for="device in availableDevices"
            :key="device.deviceId"
            :value="device.deviceId"
          >
            {{ device.label || `카메라 ${device.deviceId.slice(0, 8)}...` }}
          </option>
        </select>

        <button
          @click="refreshDevices"
          class="refresh-btn"
          :disabled="isStreamActive"
        >
          장치 새로고침
        </button>
      </div>

      <div v-if="deviceError" class="error-message">
        {{ deviceError }}
      </div>
    </div>

    <!-- 분석 결과가 있을 때 -->
    <div v-if="analysisResult" class="analysis-section">
      <div class="section-header">
        <div class="icon-wrapper">📊</div>
        <h3>분석 결과</h3>
      </div>
      <AnalysisResult
        :result="analysisResult"
        :loading="isAnalyzing"
        :error="analysisError"
        @retry="analyzeImage"
        @new-analysis="resetAnalysis"
        @save-result="saveAnalysisResult"
      />
    </div>

    <!-- 메인 콘텐츠 영역 -->
    <div v-else class="main-content">
      <!-- 카메라 화면 섹션 -->
      <div class="webcam-section">
        <div class="section-header">
          <div class="icon-wrapper">📹</div>
          <h3>카메라 화면</h3>
        </div>
        <div class="webcam-wrapper">
          <video
            ref="videoRef"
            autoplay
            playsinline
            class="webcam-video"
            :class="{ 'hidden': !isStreamActive }"
          ></video>

          <div v-if="!isStreamActive && selectedDeviceId" class="webcam-placeholder">
            <p>선택된 카메라를 시작하려면 아래 버튼을 클릭하세요</p>
            <button @click="startUSBWebcam" class="start-btn">카메라 시작</button>
          </div>

          <div v-if="!selectedDeviceId" class="no-device">
            <p>사용 가능한 카메라가 없습니다.</p>
            <p>카메라를 연결하고 "장치 새로고침" 버튼을 클릭하세요.</p>
          </div>

          <canvas ref="canvasRef" class="capture-canvas" style="display: none;"></canvas>
        </div>

        <!-- 컨트롤 버튼들 -->
        <div class="controls">
          <button
            @click="captureAndAnalyze"
            :disabled="!isStreamActive || isAnalyzing"
            class="capture-btn"
          >
            <span v-if="isAnalyzing">분석 중...</span>
            <span v-else>촬영 및 진단</span>
          </button>

          <button
            @click="stopUSBWebcam"
            :disabled="!isStreamActive"
            class="stop-btn"
          >
            카메라 중지
          </button>
        </div>
      </div>

      <!-- 촬영된 이미지 섹션 -->
      <div class="captured-section">
        <div class="section-header">
          <div class="icon-wrapper">📸</div>
          <h3>촬영된 이미지</h3>
        </div>
        <div class="captured-image-container">
          <div v-if="capturedImage" class="captured-image">
            <img :src="capturedImage" alt="촬영된 이미지" class="captured-img" />
            <div class="image-actions">
              <button @click="downloadImage" class="download-btn">이미지 저장</button>
              <button @click="clearImage" class="clear-btn">이미지 지우기</button>
              <button
                @click="analyzeImage"
                :disabled="isAnalyzing"
                class="analyze-btn"
              >
                <span v-if="isAnalyzing">분석 중...</span>
                <span v-else>다시 분석</span>
              </button>
            </div>
          </div>

          <div v-else class="no-image">
            <p>촬영된 이미지가 없습니다.</p>
            <p>카메라를 시작하고 "촬영 및 진단" 버튼을 클릭하세요.</p>
          </div>
        </div>
      </div>
    </div>

    <!-- 사용 가이드 -->
    <div class="usage-guide">
      <div class="section-header">
        <div class="icon-wrapper">ℹ️</div>
        <h3>사용 방법</h3>
      </div>
      <div class="guide-steps">
        <div class="step">
          <div class="step-number">1</div>
          <div class="step-content">
            <h4>카메라 연결</h4>
            <p>카메라를 연결하고 장치를 선택하세요</p>
          </div>
        </div>
        <div class="step">
          <div class="step-number">2</div>
          <div class="step-content">
            <h4>작물 촬영</h4>
            <p>진단하고 싶은 작물의 잎이나 과실을 명확하게 촬영하세요</p>
          </div>
        </div>
        <div class="step">
          <div class="step-number">3</div>
          <div class="step-content">
            <h4>AI 진단</h4>
            <p>AI가 자동으로 질병을 분석하고 결과를 제공합니다</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import AnalysisResult from './AnalysisResult.vue'
import { apiService, type PredictResponse } from '../services/api'

interface MediaDeviceInfo {
  deviceId: string
  kind: string
  label: string
}

const videoRef = ref<HTMLVideoElement>()
const canvasRef = ref<HTMLCanvasElement>()
const isStreamActive = ref(false)
const capturedImage = ref<string>('')
const mediaStream = ref<MediaStream | null>(null)
const availableDevices = ref<MediaDeviceInfo[]>([])
const selectedDeviceId = ref<string>('')
const deviceError = ref<string>('')
const selectedDeviceLabel = ref<string>('')

// AI 분석 관련 상태
const analysisResult = ref<PredictResponse | null>(null)
const isAnalyzing = ref(false)
const analysisError = ref<string | null>(null)

// 사용 가능한 모든 비디오 장치를 가져오기
const getAvailableDevices = async () => {
  try {
    // 먼저 카메라 권한을 요청
    await navigator.mediaDevices.getUserMedia({ video: true })

    // 모든 미디어 장치를 열거
    const devices = await navigator.mediaDevices.enumerateDevices()

    // 비디오 입력 장치만 필터링
    const videoDevices = devices.filter(device => device.kind === 'videoinput')

    availableDevices.value = videoDevices
    deviceError.value = ''

    console.log('사용 가능한 비디오 장치:', videoDevices)
  } catch (error) {
    console.error('장치 목록 가져오기 오류:', error)
    deviceError.value = '카메라 권한을 허용해주세요.'
  }
}

// 장치 새로고침
const refreshDevices = async () => {
  await getAvailableDevices()
}

// 장치 변경 시 처리
const onDeviceChange = () => {
  if (isStreamActive.value) {
    stopUSBWebcam()
  }

  const selectedDevice = availableDevices.value.find(
    device => device.deviceId === selectedDeviceId.value
  )
  selectedDeviceLabel.value = selectedDevice?.label || '알 수 없는 장치'
}

// 카메라 시작
const startUSBWebcam = async () => {
  if (!selectedDeviceId.value) {
    deviceError.value = '카메라 장치를 선택해주세요.'
    return
  }

  try {
    const constraints = {
      video: {
        deviceId: { exact: selectedDeviceId.value },
        width: { ideal: 1920 },
        height: { ideal: 1080 }
      }
    }

    const stream = await navigator.mediaDevices.getUserMedia(constraints)

    if (videoRef.value) {
      videoRef.value.srcObject = stream
      mediaStream.value = stream
      isStreamActive.value = true
      deviceError.value = ''

      console.log('카메라 시작됨:', selectedDeviceLabel.value)
    }
  } catch (error) {
    console.error('카메라 시작 오류:', error)
    deviceError.value = `선택된 카메라를 시작할 수 없습니다: ${error}`
  }
}

// 카메라 중지
const stopUSBWebcam = () => {
  if (mediaStream.value) {
    mediaStream.value.getTracks().forEach(track => track.stop())
    mediaStream.value = null
  }

  if (videoRef.value) {
    videoRef.value.srcObject = null
  }

  isStreamActive.value = false
  console.log('카메라 중지됨')
}

// 이미지 촬영
const captureImage = () => {
  if (!videoRef.value || !canvasRef.value) return

  const video = videoRef.value
  const canvas = canvasRef.value
  const context = canvas.getContext('2d')

  if (!context) return

  // 비디오 크기에 맞춰 캔버스 크기 설정
  canvas.width = video.videoWidth
  canvas.height = video.videoHeight

  // 비디오 프레임을 캔버스에 그리기
  context.drawImage(video, 0, 0, canvas.width, canvas.height)

  // 캔버스를 이미지로 변환
  capturedImage.value = canvas.toDataURL('image/jpeg', 0.9)

  console.log('이미지 촬영 완료:', canvas.width, 'x', canvas.height)
}

// 이미지 다운로드
const downloadImage = () => {
  if (!capturedImage.value) return

  const link = document.createElement('a')
  link.download = `usb-webcam-capture-${new Date().getTime()}.jpg`
  link.href = capturedImage.value
  link.click()
}

// 이미지 지우기
const clearImage = () => {
  capturedImage.value = ''
  resetAnalysis()
}

// AI 이미지 분석
const analyzeImage = async () => {
  if (!capturedImage.value) {
    analysisError.value = '분석할 이미지가 없습니다.'
    return
  }

  isAnalyzing.value = true
  analysisError.value = null

  try {
    // Base64 이미지를 File 객체로 변환
    const response = await fetch(capturedImage.value)
    const blob = await response.blob()
    const file = new File([blob], 'captured-image.jpg', { type: 'image/jpeg' })

    // AI 모델로 예측
    const result = await apiService.predictImage(file)
    analysisResult.value = result
    console.log('AI 분석 완료:', result)
  } catch (error) {
    console.error('AI 분석 오류:', error)
    analysisError.value = '이미지 분석 중 오류가 발생했습니다. 다시 시도해주세요.'
  } finally {
    isAnalyzing.value = false
  }
}

// 촬영 및 분석
const captureAndAnalyze = async () => {
  captureImage()
  // 이미지 촬영 후 잠시 대기 후 분석 시작
  setTimeout(() => {
    analyzeImage()
  }, 500)
}

// 분석 결과 초기화
const resetAnalysis = () => {
  analysisResult.value = null
  analysisError.value = null
}

// 분석 결과 저장
const saveAnalysisResult = async () => {
  if (!analysisResult.value || !capturedImage.value) return

  try {
    // 이미 analyzeImage에서 저장되므로 현재 결과만 로그
    console.log('분석 결과 저장 완료:', analysisResult.value)

    // 성공 메시지 표시 (실제로는 토스트나 알림을 사용할 수 있음)
    alert('분석 결과가 성공적으로 저장되었습니다.')
  } catch (error) {
    console.error('결과 저장 오류:', error)
    alert('결과 저장 중 오류가 발생했습니다.')
  }
}

onMounted(async () => {
  // 컴포넌트 마운트 시 사용 가능한 장치 목록 가져오기
  await getAvailableDevices()
})

onUnmounted(() => {
  // 컴포넌트 언마운트 시 카메라 정리
  stopUSBWebcam()
})
</script>

<style scoped>
/* 기존 스타일에 추가 */
.usb-webcam-container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 20px;
  text-align: center;
  color: #000000;
  /* AgroEye 헤더와 같은 파란색-보라색 그라데이션으로 변경 */
  background: linear-gradient(135deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%);
  min-height: 100vh;
  position: relative;
}

/* 배경 패턴 추가 */
.usb-webcam-container::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-image:
    radial-gradient(circle at 25% 25%, rgba(255,255,255,0.1) 1px, transparent 1px),
    radial-gradient(circle at 75% 75%, rgba(255,255,255,0.1) 1px, transparent 1px);
  background-size: 50px 50px;
  pointer-events: none;
}

/* 컨테이너 내용을 배경 위로 */
.usb-webcam-container > * {
  position: relative;
  z-index: 1;
}

/* 카드 공통 스타일 */
.device-selection,
.webcam-section,
.captured-section,
.usage-guide {
  background: rgb(230, 252, 212);
  border-radius: 16px;
  box-shadow:
    0 8px 32px rgba(0, 0, 0, 0.1),
    0 2px 8px rgba(0, 0, 0, 0.05);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

/* 카드 호버 효과 */
.device-selection:hover,
.webcam-section:hover,
.captured-section:hover,
.usage-guide:hover {
  transform: translateY(-4px);
  box-shadow:
    0 12px 40px rgba(0, 0, 0, 0.15),
    0 4px 12px rgba(0, 0, 0, 0.1);
}

/* 카드 상단 장식 */
.device-selection::before,
.webcam-section::before,
.captured-section::before,
.usage-guide::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
  border-radius: 16px 16px 0 0;
}

/* 섹션 헤더 스타일 */
.section-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 20px;
  padding-bottom: 15px;
  border-bottom: 2px solid rgba(102, 126, 234, 0.1);
}

.icon-wrapper {
  width: 40px;
  height: 40px;
  background: linear-gradient(135deg, #667eea, #764ba2);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  color: white;
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}

.section-header h3 {
  margin: 0;
  color: #2d3748;
  font-size: 20px;
  font-weight: 600;
}

/* 버튼 공통 스타일 */
button {
  border: none;
  border-radius: 8px;
  padding: 12px 24px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

/* 버튼 호버 효과 */
button:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
}

/* 주요 액션 버튼 */
.capture-btn {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}

.start-btn {
  background: linear-gradient(135deg, #28a745, #20c997);
  color: white;
  box-shadow: 0 4px 15px rgba(40, 167, 69, 0.4);
}

.stop-btn {
  background: linear-gradient(135deg, #dc3545, #e74c3c);
  color: white;
  box-shadow: 0 4px 15px rgba(220, 53, 69, 0.4);
}

/* 보조 버튼 */
.download-btn {
  background: linear-gradient(135deg, #ffc107, #ffb300);
  color: #212529;
  box-shadow: 0 4px 15px rgba(255, 193, 7, 0.4);
}

.clear-btn {
  background: linear-gradient(135deg, #6c757d, #5a6268);
  color: white;
  box-shadow: 0 4px 15px rgba(108, 117, 125, 0.4);
}

.analyze-btn {
  background: linear-gradient(135deg, #17a2b8, #138496);
  color: white;
  box-shadow: 0 4px 15px rgba(23, 162, 184, 0.4);
}

.refresh-btn {
  background: linear-gradient(135deg, #6c757d, #5a6268);
  color: white;
  box-shadow: 0 4px 15px rgba(108, 117, 125, 0.4);
}

/* 사용 가이드 스타일 개선 */
.guide-steps {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 24px;
  margin-top: 30px;
}

.step {
  background: rgba(255, 255, 255, 0.8);
  border-radius: 12px;
  padding: 24px;
  border: 1px solid rgba(102, 126, 234, 0.1);
  transition: all 0.3s ease;
  position: relative;
}

.step:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
  background: rgba(255, 255, 255, 0.95);
}

.step-number {
  width: 50px;
  height: 50px;
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  font-size: 20px;
  margin-bottom: 16px;
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}

.step-content h4 {
  margin: 0 0 12px 0;
  font-size: 18px;
  color: #2d3748;
  font-weight: 600;
}

.step-content p {
  margin: 0;
  font-size: 14px;
  line-height: 1.6;
  color: #4a5568;
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

.device-selection,
.webcam-section,
.captured-section,
.usage-guide {
  animation: fadeInUp 0.6s ease-out;
}

/* 순차적 애니메이션 */
.device-selection { animation-delay: 0.1s; }
.webcam-section { animation-delay: 0.2s; }
.captured-section { animation-delay: 0.3s; }
.usage-guide { animation-delay: 0.4s; }

/* 로딩 애니메이션 개선 */
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.7; }
}

.webcam-placeholder,
.no-device,
.no-image {
  animation: pulse 2s infinite;
}

/* 기존 스타일 유지하면서 개선 */
.device-selection {
  margin-bottom: 30px;
  padding: 20px;
}

.device-list {
  display: flex;
  gap: 10px;
  justify-content: center;
  align-items: center;
  flex-wrap: wrap;
  margin-top: 15px;
}

.device-select {
  padding: 10px 15px;
  border: 1px solid #ddd;
  border-radius: 4px;
  min-width: 300px;
  background-color: white;
}

.error-message {
  color: #dc3545;
  margin-top: 10px;
  font-weight: 500;
}

.main-content {
  display: flex;
  gap: 30px;
  flex-wrap: wrap;
  justify-content: center;
  margin-top: 20px;
}

.webcam-section, .captured-section {
  flex: 1;
  min-width: 500px;
  max-width: 650px;
  padding: 30px;
}

.webcam-section h3, .captured-section h3 {
  margin-bottom: 15px;
  text-align: left;
}

.webcam-wrapper {
  position: relative;
  margin-bottom: 25px;
  border-radius: 12px;
  overflow: hidden;
  background-color: rgba(255, 255, 255, 0.95);
  min-height: 400px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.webcam-video {
  width: 100%;
  max-width: 100%;
  height: auto;
  border-radius: 8px;
}

.webcam-video.hidden {
  display: none;
}

.webcam-placeholder, .no-device {
  padding: 40px;
  color: #000000;
}

.controls {
  display: flex;
  gap: 10px;
  justify-content: center;
  flex-wrap: wrap;
}

.captured-image-container {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.captured-image {
  margin-bottom: 25px;
  padding: 25px;
  border: 1px solid #ddd;
  border-radius: 12px;
  background-color: #f9f9f9;
  width: 100%;
  max-width: 550px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.captured-img {
  max-width: 100%;
  max-height: 450px;
  border-radius: 8px;
  margin: 15px 0;
}

.image-actions {
  display: flex;
  gap: 10px;
  justify-content: center;
  margin-top: 15px;
}

.no-image {
  padding: 40px;
  color: #000000;
}

.analysis-section {
  margin-top: 30px;
}

.usage-guide {
  margin-top: 40px;
  padding: 30px;
}

.usage-guide h3 {
  text-align: center;
  margin-bottom: 25px;
}

h2 {
  margin-bottom: 10px;
  font-size: 28px;
  color: #ffffff;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
}

.subtitle {
  margin-bottom: 30px;
  font-size: 16px;
  color: #ffffff;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
}

h3 {
  margin-bottom: 15px;
  color: #000000;
}

@media (max-width: 1200px) {
  .main-content {
    flex-direction: column;
    align-items: center;
  }

  .webcam-section, .captured-section {
    width: 100%;
    max-width: 100%;
    min-width: auto;
  }

  .image-actions {
    flex-direction: column;
    align-items: center;
  }
}

@media (max-width: 768px) {
  /* 섹션 전체를 뷰포트 가로로 확장 (full-bleed) */
  .usb-webcam-container {
    width: 100vw;
    margin-left: calc(50% - 50vw);
    margin-right: calc(50% - 50vw);
    padding-left: 8px;
    padding-right: 8px;
    border-radius: 0;
  }

  /* 카드 폭을 가득 채우고, 반응형 모서리로 조정 */
  .device-selection,
  .webcam-section,
  .captured-section,
  .usage-guide {
    width: 100%;
    max-width: 100%;
    min-width: 0;
    margin-left: 0;
    margin-right: 0;
    border-radius: 10px;
  }

  /* 사용 방법 그리드가 부모 폭을 꽉 사용 */
  .guide-steps {
    width: 100%;
  }
}

@media (max-width: 480px) {
  .usb-webcam-container {
    padding: 12px;
  }

  h2 {
    font-size: 20px;
  }

  .subtitle {
    font-size: 13px;
  }

  .device-select {
    padding: 6px 10px;
  }

  .image-actions button {
    padding: 8px 12px;
  }

  .usage-guide {
    padding: 16px;
  }

  .step {
    gap: 10px;
  }

  .step-number {
    width: 32px;
    height: 32px;
    font-size: 16px;
  }

  .step-content h4 {
    font-size: 14px;
  }
}
</style>

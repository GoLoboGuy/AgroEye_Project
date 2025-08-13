<template>
  <div class="usb-webcam-container">
    <h2>🌱 식물 병충해 진단 시스템</h2>
    <p class="subtitle">웹캠으로 작물을 촬영하여 AI가 병충해를 진단해드립니다</p>

    <!-- 웹캠 선택 섹션 -->
    <div class="device-selection">
      <h3>웹캠 장치 선택</h3>
      <div class="device-list">
        <select
          v-model="selectedDeviceId"
          @change="onDeviceChange"
          class="device-select"
          :disabled="isStreamActive"
        >
          <option value="">웹캠 장치를 선택하세요</option>
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
      <!-- 웹캠 화면 섹션 -->
      <div class="webcam-section">
        <h3>웹캠 화면</h3>
        <div class="webcam-wrapper">
          <video
            ref="videoRef"
            autoplay
            playsinline
            class="webcam-video"
            :class="{ 'hidden': !isStreamActive }"
          ></video>

          <div v-if="!isStreamActive && selectedDeviceId" class="webcam-placeholder">
            <p>선택된 USB 웹캠을 시작하려면 아래 버튼을 클릭하세요</p>
            <button @click="startUSBWebcam" class="start-btn">USB 웹캠 시작</button>
          </div>

          <div v-if="!selectedDeviceId" class="no-device">
            <p>사용 가능한 USB 웹캠이 없습니다.</p>
            <p>USB 웹캠을 연결하고 "장치 새로고침" 버튼을 클릭하세요.</p>
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
            웹캠 중지
          </button>
        </div>
      </div>

      <!-- 촬영된 이미지 섹션 -->
      <div class="captured-section">
        <h3>촬영된 이미지</h3>
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
            <p>웹캠을 시작하고 "촬영 및 진단" 버튼을 클릭하세요.</p>
          </div>
        </div>
      </div>
    </div>

    <!-- 사용 가이드 -->
    <div class="usage-guide">
      <h3>사용 방법</h3>
      <div class="guide-steps">
        <div class="step">
          <div class="step-number">1</div>
          <div class="step-content">
            <h4>웹캠 연결</h4>
            <p>USB 웹캠을 연결하고 장치를 선택하세요</p>
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
            <p>AI가 자동으로 병충해를 분석하고 결과를 제공합니다</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import AnalysisResult from './AnalysisResult.vue'
import { apiService, type ModelPrediction } from '../services/api'

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
const analysisResult = ref<ModelPrediction | null>(null)
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

// USB 웹캠 시작
const startUSBWebcam = async () => {
  if (!selectedDeviceId.value) {
    deviceError.value = '웹캠 장치를 선택해주세요.'
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

      console.log('USB 웹캠 시작됨:', selectedDeviceLabel.value)
    }
  } catch (error) {
    console.error('USB 웹캠 시작 오류:', error)
    deviceError.value = `선택된 USB 웹캠을 시작할 수 없습니다: ${error}`
  }
}

// USB 웹캠 중지
const stopUSBWebcam = () => {
  if (mediaStream.value) {
    mediaStream.value.getTracks().forEach(track => track.stop())
    mediaStream.value = null
  }

  if (videoRef.value) {
    videoRef.value.srcObject = null
  }

  isStreamActive.value = false
  console.log('USB 웹캠 중지됨')
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
    // Base64 이미지를 File 객체로 변환
    const response = await fetch(capturedImage.value)
    const blob = await response.blob()
    const file = new File([blob], 'captured-image.jpg', { type: 'image/jpeg' })

    // Backend에 저장
    const results = await apiService.uploadAndAnalyzeImage(file)
    console.log('분석 결과 저장 완료:', results)

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
  // 컴포넌트 언마운트 시 웹캠 정리
  stopUSBWebcam()
})
</script>

<style scoped>
.usb-webcam-container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 20px;
  text-align: center;
}

.device-selection {
  margin-bottom: 30px;
  padding: 20px;
  background-color: #f8f9fa;
  border-radius: 8px;
  border: 1px solid #e9ecef;
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
  font-size: 14px;
  min-width: 300px;
  background-color: white;
}

.refresh-btn {
  background-color: #6c757d;
  color: white;
  border: none;
  padding: 10px 15px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

.refresh-btn:hover:not(:disabled) {
  background-color: #5a6268;
}

.refresh-btn:disabled {
  background-color: #ccc;
  cursor: not-allowed;
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
  background-color: #f5f5f5;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.webcam-section h3, .captured-section h3 {
  color: #333;
  margin-bottom: 15px;
  text-align: left;
}

.webcam-wrapper {
  position: relative;
  margin-bottom: 25px;
  border-radius: 12px;
  overflow: hidden;
  background-color: #f5f5f5;
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
  color: #666;
}

.start-btn {
  background-color: #28a745;
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 16px;
  margin-top: 10px;
}

.start-btn:hover {
  background-color: #218838;
}

.controls {
  display: flex;
  gap: 10px;
  justify-content: center;
  flex-wrap: wrap;
}

.capture-btn, .stop-btn {
  padding: 12px 24px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 16px;
  font-weight: 500;
}

.capture-btn {
  background-color: #007bff;
  color: white;
}

.capture-btn:hover:not(:disabled) {
  background-color: #0056b3;
}

.capture-btn:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

.stop-btn {
  background-color: #dc3545;
  color: white;
}

.stop-btn:hover:not(:disabled) {
  background-color: #c82333;
}

.stop-btn:disabled {
  background-color: #ccc;
  cursor: not-allowed;
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

.download-btn, .clear-btn {
  padding: 10px 20px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

.download-btn {
  background-color: #ffc107;
  color: #212529;
}

.download-btn:hover {
  background-color: #e0a800;
}

.clear-btn {
  background-color: #6c757d;
  color: white;
}

.clear-btn:hover {
  background-color: #5a6268;
}

.analyze-btn {
  background-color: #17a2b8;
  color: white;
}

.analyze-btn:hover:not(:disabled) {
  background-color: #138496;
}

.analyze-btn:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

.no-image {
  padding: 40px;
  color: #666;
}

.analysis-section {
  margin-top: 30px;
}

.usage-guide {
  margin-top: 40px;
  padding: 30px;
  background-color: #f8f9fa;
  border-radius: 12px;
  border: 1px solid #e9ecef;
}

.usage-guide h3 {
  text-align: center;
  margin-bottom: 25px;
  color: #333;
}

.guide-steps {
  display: flex;
  gap: 30px;
  justify-content: center;
  flex-wrap: wrap;
}

.step {
  display: flex;
  align-items: flex-start;
  gap: 15px;
  max-width: 250px;
}

.step-number {
  width: 40px;
  height: 40px;
  background-color: #007bff;
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  font-size: 18px;
  flex-shrink: 0;
}

.step-content h4 {
  margin: 0 0 8px 0;
  color: #333;
  font-size: 16px;
}

.step-content p {
  margin: 0;
  color: #666;
  font-size: 14px;
  line-height: 1.5;
}

h2 {
  color: #333;
  margin-bottom: 10px;
  font-size: 28px;
}

.subtitle {
  color: #666;
  margin-bottom: 30px;
  font-size: 16px;
}

h3 {
  color: #555;
  margin-bottom: 15px;
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
  .usb-webcam-container {
    padding: 16px;
  }

  h2 {
    font-size: 24px;
  }

  .subtitle {
    font-size: 14px;
  }

  .guide-steps {
    flex-direction: column;
    align-items: center;
  }

  .step {
    max-width: 100%;
  }

  .usage-guide {
    padding: 20px;
  }

  .device-list {
    flex-direction: column;
    align-items: center;
  }

  .device-select {
    min-width: 250px;
  }
}
</style>

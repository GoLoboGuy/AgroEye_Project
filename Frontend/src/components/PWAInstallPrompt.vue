<template>
  <div v-if="showInstallPrompt" class="pwa-install-prompt">
    <div class="pwa-install-content">
      <div class="pwa-install-icon">
        <svg width="48" height="48" viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg">
          <circle cx="256" cy="256" r="240" fill="#4CAF50" stroke="#2E7D32" stroke-width="8"/>
          <path d="M 200 180 Q 180 160 160 180 Q 140 200 160 220 Q 180 240 200 220 Q 220 200 200 180 Z" fill="#8BC34A"/>
          <path d="M 240 160 Q 220 140 200 160 Q 180 180 200 200 Q 220 220 240 200 Q 260 180 240 160 Z" fill="#8BC34A"/>
          <path d="M 280 180 Q 260 160 240 180 Q 220 200 240 220 Q 260 240 280 220 Q 300 200 280 180 Z" fill="#8BC34A"/>
          <path d="M 320 200 Q 300 180 280 200 Q 260 220 280 240 Q 300 260 320 240 Q 340 220 320 200 Z" fill="#8BC34A"/>
          <path d="M 240 240 L 240 320" stroke="#2E7D32" stroke-width="6" fill="none"/>
          <rect x="180" y="280" width="120" height="80" rx="15" fill="#424242"/>
          <circle cx="240" cy="320" r="25" fill="#212121"/>
          <circle cx="240" cy="320" r="20" fill="#000000"/>
          <circle cx="240" cy="320" r="15" fill="#1E88E5"/>
          <circle cx="280" cy="300" r="8" fill="#FFC107"/>
          <rect x="290" y="310" width="15" height="8" rx="2" fill="#757575"/>
        </svg>
      </div>
      <div class="pwa-install-text">
        <h3>AgroEye 앱 설치</h3>
        <p>홈 화면에 추가하여 더 빠르게 접근하세요!</p>
      </div>
      <div class="pwa-install-buttons">
        <button @click="installPWA" class="pwa-install-btn">
          설치하기
        </button>
        <button @click="dismissPrompt" class="pwa-dismiss-btn">
          나중에
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'

// PWA 설치 프롬프트 타입 정의
interface BeforeInstallPromptEvent extends Event {
  preventDefault(): void
  prompt(): Promise<{ outcome: 'accepted' | 'dismissed' }>
  userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>
}

const showInstallPrompt = ref(false)
let deferredPrompt: BeforeInstallPromptEvent | null = null

onMounted(() => {
  // 모바일 환경 감지
  const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)

  // PWA 설치 이벤트 리스너
  window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault()
    deferredPrompt = e as BeforeInstallPromptEvent
    showInstallPrompt.value = true
  })

  // PWA가 이미 설치되었는지 확인
  if (window.matchMedia('(display-mode: standalone)').matches) {
    showInstallPrompt.value = false
  }

  // 모바일에서 PWA 설치 가능 여부 확인 및 수동 프롬프트 표시
  if (isMobile && !window.matchMedia('(display-mode: standalone)').matches) {
    // 3초 후에 수동 설치 안내 표시
    setTimeout(() => {
      if (!showInstallPrompt.value) {
        showInstallPrompt.value = true
      }
    }, 3000)
  }
})

const installPWA = async () => {
  if (deferredPrompt) {
    deferredPrompt.prompt()
    const { outcome } = await deferredPrompt.userChoice
    if (outcome === 'accepted') {
      console.log('PWA 설치됨')
    }
    deferredPrompt = null
    showInstallPrompt.value = false
  } else {
    // 수동 설치 안내 (iOS Safari 등)
    showManualInstallGuide()
  }
}

const showManualInstallGuide = () => {
  const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent)
  const isAndroid = /Android/.test(navigator.userAgent)

  let message = ''
  if (isIOS) {
    message = 'Safari에서 공유 버튼(□)을 탭하고 "홈 화면에 추가"를 선택하세요.'
  } else if (isAndroid) {
    message = 'Chrome 메뉴(⋮)에서 "홈 화면에 추가"를 선택하세요.'
  } else {
    message = '브라우저 메뉴에서 "홈 화면에 추가" 또는 "앱 설치"를 선택하세요.'
  }

  alert(message)
  showInstallPrompt.value = false
}

const dismissPrompt = () => {
  showInstallPrompt.value = false
}
</script>

<style scoped>
/* PWA 공통 스타일 */
.pwa-prompt {
  position: fixed;
  background: rgba(255, 255, 255, 0.95);
  border-radius: 16px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1), 0 2px 8px rgba(0, 0, 0, 0.05);
  padding: 20px;
  z-index: 1000;
  border: 1px solid rgba(255, 255, 255, 0.2);
  backdrop-filter: blur(10px);
  transition: all 0.3s ease;
  animation: fadeInUp 0.4s ease-out;
}

.pwa-prompt::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
  border-radius: 16px 16px 0 0;
}

.pwa-install-prompt {
  @apply pwa-prompt;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  max-width: 320px;
}

.pwa-install-prompt:hover {
  transform: translateX(-50%) translateY(-2px);
}

/* PWA 애니메이션 */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateX(-50%) translateY(20px); }
  to { opacity: 1; transform: translateX(-50%) translateY(0); }
}

.pwa-install-content {
  display: flex;
  align-items: center;
  gap: 12px;
}

.pwa-install-icon {
  flex-shrink: 0;
}

.pwa-install-text {
  flex: 1;
}

.pwa-install-text h3 {
  margin: 0 0 4px 0;
  font-size: 16px;
  font-weight: 600;
  color: #2d3748;
}

.pwa-install-text p {
  margin: 0;
  font-size: 14px;
  line-height: 1.4;
  color: #4a5568;
}

.pwa-install-buttons {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 12px;
}

/* PWA 버튼 스타일 */
.pwa-install-buttons button {
  border: none;
  padding: 10px 16px;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
  color: white;
}

.pwa-install-btn {
  background: linear-gradient(135deg, #28a745, #20c997);
}

.pwa-dismiss-btn {
  background: linear-gradient(135deg, #6c757d, #5a6268);
}

.pwa-install-buttons button:hover {
  transform: translateY(-1px);
  opacity: 0.9;
}

@media (max-width: 480px) {
  .pwa-install-prompt {
    bottom: 10px;
    left: 10px;
    right: 10px;
    transform: none;
    max-width: none;
  }

  .pwa-install-prompt:hover {
    transform: translateY(-2px);
  }

  @keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .pwa-install-content {
    flex-direction: column;
    text-align: center;
  }

  .pwa-install-buttons {
    flex-direction: row;
    justify-content: center;
  }
}
</style>

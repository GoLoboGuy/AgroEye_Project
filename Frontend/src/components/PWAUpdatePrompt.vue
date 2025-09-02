<template>
  <div v-if="showUpdatePrompt" class="pwa-update-prompt">
    <div class="pwa-update-content">
      <div class="pwa-update-icon">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M12 2L13.09 8.26L20 9L13.09 9.74L12 16L10.91 9.74L4 9L10.91 8.26L12 2Z" fill="#FFC107"/>
        </svg>
      </div>
      <div class="pwa-update-text">
        <h4>새로운 업데이트가 있습니다</h4>
        <p>최신 기능을 사용하려면 앱을 새로고침하세요</p>
      </div>
      <div class="pwa-update-buttons">
        <button @click="updateApp" class="pwa-update-btn">
          새로고침
        </button>
        <button @click="dismissUpdate" class="pwa-dismiss-btn">
          나중에
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'

const showUpdatePrompt = ref(false)

onMounted(() => {
  // PWA 업데이트 이벤트 리스너
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.addEventListener('controllerchange', () => {
      showUpdatePrompt.value = true
    })
  }
})

const updateApp = () => {
  window.location.reload()
}

const dismissUpdate = () => {
  showUpdatePrompt.value = false
}
</script>

<style scoped>
.pwa-update-prompt {
  @apply pwa-prompt;
  top: 20px;
  right: 20px;
  padding: 16px;
  max-width: 300px;
}

.pwa-update-prompt:hover {
  transform: translateY(-2px);
}

/* PWA 공통 스타일 (상단에서 사용) */
.pwa-prompt {
  position: fixed;
  background: rgba(255, 255, 255, 0.95);
  border-radius: 16px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1), 0 2px 8px rgba(0, 0, 0, 0.05);
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

@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(-20px); }
  to { opacity: 1; transform: translateY(0); }
}

.pwa-update-content {
  display: flex;
  align-items: center;
  gap: 8px;
}

.pwa-update-icon {
  flex-shrink: 0;
}

.pwa-update-text {
  flex: 1;
}

.pwa-update-text h4 {
  margin: 0 0 2px 0;
  font-size: 14px;
  font-weight: 600;
  color: #2d3748;
}

.pwa-update-text p {
  margin: 0;
  font-size: 12px;
  line-height: 1.3;
  color: #4a5568;
}

.pwa-update-buttons {
  display: flex;
  gap: 6px;
  margin-top: 8px;
}

.pwa-update-buttons button {
  border: none;
  padding: 6px 12px;
  border-radius: 8px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
}

.pwa-update-btn {
  background: linear-gradient(135deg, #ffc107, #ffb300);
  color: #212529;
}

.pwa-dismiss-btn {
  background: linear-gradient(135deg, #6c757d, #5a6268);
  color: white;
}

.pwa-update-buttons button:hover {
  transform: translateY(-1px);
  opacity: 0.9;
}

@media (max-width: 480px) {
  .pwa-update-prompt {
    top: 10px;
    right: 10px;
    left: 10px;
    max-width: none;
  }

  @keyframes fadeInUp {
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
  }
}
</style>

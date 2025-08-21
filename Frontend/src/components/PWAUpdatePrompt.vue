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
  position: fixed;
  top: 20px;
  right: 20px;
  background: rgba(255, 255, 255, 0.98);
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
  padding: 12px;
  max-width: 300px;
  z-index: 1000;
  border: 1px solid rgba(0, 0, 0, 0.1);
  backdrop-filter: blur(10px);
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
}

.pwa-update-text p {
  margin: 0;
  font-size: 12px;
  line-height: 1.3;
}

.pwa-update-buttons {
  display: flex;
  gap: 6px;
  margin-top: 8px;
}

.pwa-update-btn {
  background: #FFC107;
  color: #333;
  border: none;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
  transition: background-color 0.2s;
}

.pwa-update-btn:hover {
  background: #FFB300;
}

.pwa-dismiss-btn {
  background: transparent;
  border: 1px solid #ddd;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
  cursor: pointer;
  transition: background-color 0.2s;
}

.pwa-dismiss-btn:hover {
  background: #f5f5f5;
}

@media (max-width: 480px) {
  .pwa-update-prompt {
    top: 10px;
    right: 10px;
    left: 10px;
    max-width: none;
  }
}
</style>

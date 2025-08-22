<script setup lang="ts">
import USBWebcamCapture from './components/USBWebcamCapture.vue'
import PWAInstallPrompt from './components/PWAInstallPrompt.vue'
import PWAUpdatePrompt from './components/PWAUpdatePrompt.vue'
</script>

<template>
  <div id="app">
    <header class="app-header">
      <div class="header-content">
        <h1>🌱 AgroEye</h1>
        <p>AI 기반 농작물 질병 진단 시스템</p>
      </div>
    </header>
    <main class="app-main">
      <USBWebcamCapture />
    </main>
    <footer class="app-footer">
      <p>&copy; 2025 AgroEye. AI 기반 농작물 질병 진단 시스템</p>
    </footer>
    <PWAInstallPrompt />
    <PWAUpdatePrompt />
  </div>
</template>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
  line-height: 1.6;
}

#app {
  width: 100%;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.app-header {
  background: linear-gradient(135deg,
    rgba(102, 126, 234, 0.9) 0%,
    rgba(118, 75, 162, 0.9) 100%
  );
  color: white;
  padding: 20px 0;
  text-align: center;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  backdrop-filter: blur(10px);
}

.header-content h1 {
  font-size: 32px;
  font-weight: 700;
  margin-bottom: 8px;
}

.header-content p {
  opacity: 0.9;
}

.app-main {
  flex: 1;
  padding: 20px 0;
}

.app-footer {
  background-color: rgba(51, 51, 51, 0.9);
  color: white;
  text-align: center;
  padding: 20px;
  margin-top: auto;
  backdrop-filter: blur(10px);
}

.app-footer p {
  opacity: 0.8;
}

@media (max-width: 768px) {
  .header-content h1 {
    font-size: 24px;
  }

  .app-main {
    padding: 16px 0;
  }
}
</style>

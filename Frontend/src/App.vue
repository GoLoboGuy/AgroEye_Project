<script setup lang="ts">
import USBWebcamCapture from './components/USBWebcamCapture.vue'
</script>

<template>
  <div id="app">
    <header class="app-header">
      <div class="header-content">
        <h1>🌱 AgroEye</h1>
        <p>AI 기반 식물 병충해 진단 시스템</p>
      </div>
    </header>
    <main class="app-main">
      <USBWebcamCapture />
    </main>
    <footer class="app-footer">
      <p>&copy; 2024 AgroEye. AI 기반 식물 병충해 진단 시스템</p>
    </footer>
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
  color: #333;
}

#app {
  width: 100%;
  min-height: 100vh;
  background-color: #f5f5f5;
  display: flex;
  flex-direction: column;
}

.app-header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 20px 0;
  text-align: center;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.header-content h1 {
  font-size: 32px;
  font-weight: 700;
  margin-bottom: 8px;
}

.header-content p {
  font-size: 16px;
  opacity: 0.9;
}

.app-main {
  flex: 1;
  padding: 20px 0;
}

.app-footer {
  background-color: #333;
  color: white;
  text-align: center;
  padding: 20px;
  margin-top: auto;
}

.app-footer p {
  opacity: 0.8;
  font-size: 14px;
}

@media (max-width: 768px) {
  .header-content h1 {
    font-size: 24px;
  }

  .header-content p {
    font-size: 14px;
  }

  .app-main {
    padding: 16px 0;
  }
}
</style>

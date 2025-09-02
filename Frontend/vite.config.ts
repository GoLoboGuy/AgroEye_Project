import { fileURLToPath, URL } from 'node:url'

import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'
// import vueDevTools from 'vite-plugin-vue-devtools'
import { VitePWA } from 'vite-plugin-pwa'

// https://vite.dev/config/
export default defineConfig(({ command, mode }) => {
  // 환경 변수 로드
  const env = loadEnv(mode, process.cwd(), '')

  return {
    server: {
      host: '0.0.0.0',
      port: 5173,
      allowedHosts: [
        'localhost',
        '127.0.0.1',
        '.ngrok-free.app', // 모든 ngrok 도메인 허용
        '.ngrok.io'        // 구버전 ngrok도 허용
      ],
      // 프록시 설정 - Backend API 요청을 프록시
      proxy: {
        '/api': {
          target: env.VITE_PROXY_TARGET || 'http://localhost:8000',
          changeOrigin: true,
          secure: false,
          configure: (proxy, options) => {
            proxy.on('proxyReq', (proxyReq, req, res) => {
              if (env.VITE_DEBUG === 'true') {
                console.log('📤 프록시 요청:', req.method, req.url, '→', options.target)
              }
            })
            proxy.on('proxyRes', (proxyRes, req, res) => {
              if (env.VITE_DEBUG === 'true') {
                console.log('📥 프록시 응답:', proxyRes.statusCode, req.url)
              }
            })
            proxy.on('error', (err, req, res) => {
              console.error('❌ 프록시 에러:', err.message)
              console.error('   요청 URL:', req.url)
              console.error('   타겟:', options.target)
            })
          }
        },
        '/health': {
          target: env.VITE_PROXY_TARGET || 'http://localhost:8000',
          changeOrigin: true,
          secure: false,
          configure: (proxy, options) => {
            proxy.on('error', (err, req, res) => {
              console.error('❌ 헬스체크 프록시 에러:', err.message)
            })
          }
        }
      }
    },
    plugins: [
      vue(),
      // vueDevTools(),
      VitePWA({
        registerType: 'autoUpdate',
        includeAssets: ['favicon.ico', 'icon.svg'],
        manifest: {
          name: 'AgroEye - 식물 병해충 탐지',
          short_name: 'AgroEye',
          description: 'AI 기반 식물 병해충 탐지 및 분석 서비스',
          theme_color: '#4CAF50',
          background_color: '#ffffff',
          display: 'standalone',
          orientation: 'portrait',
          scope: '/',
          start_url: '/',
          prefer_related_applications: false,
          categories: ['utilities', 'productivity', 'education'],
          lang: 'ko',
          dir: 'ltr',
          icons: [
            {
              src: 'icon-192.png',
              sizes: '192x192',
              type: 'image/png',
              purpose: 'any'
            },
            {
              src: 'icon-512.png',
              sizes: '512x512',
              type: 'image/png',
              purpose: 'any'
            },
            {
              src: 'icon-512.png',
              sizes: '512x512',
              type: 'image/png',
              purpose: 'maskable'
            }
          ],
          screenshots: [
            {
              src: 'screenshot-wide.png',
              sizes: '1280x720',
              type: 'image/png',
              form_factor: 'wide'
            },
            {
              src: 'screenshot-narrow.png',
              sizes: '750x1334',
              type: 'image/png',
              form_factor: 'narrow'
            }
          ]
        },
        workbox: {
          globPatterns: ['**/*.{js,css,html,ico,png,svg}'],
          runtimeCaching: [
            {
              urlPattern: /^https:\/\/api\./,
              handler: 'NetworkFirst',
              options: {
                cacheName: 'api-cache',
                expiration: {
                  maxEntries: 100,
                  maxAgeSeconds: 60 * 60 * 24 // 24시간
                }
              }
            }
          ]
        }
      })
    ],
    resolve: {
      alias: {
        '@': fileURLToPath(new URL('./src', import.meta.url))
      },
    },
  }
})

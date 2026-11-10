import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // The browser only talks to this origin: /api/... is forwarded to the FastAPI
    // service (with the /api prefix removed), so no CORS configuration is needed.
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        rewrite: (path: string) => path.replace(/^\/api/, ''),
      },
    },
  },
})

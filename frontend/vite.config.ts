import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Optional dev proxy: same-origin requests to /api/* → FastAPI (use if you prefer not relying on CORS in dev).
// Default: SPA calls VITE_API_BASE_URL directly (CORS enabled on backend).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});

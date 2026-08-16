import { defineConfig } from 'vite';
import { alphaTab } from '@coderline/alphatab-vite';
import { resolve } from 'node:path';

// alphaTab loads its music font (Bravura), a soundfont, a web worker for
// layout, and an audio worklet for the synth -- none of which a bundler can
// discover by following imports. The official plugin wires all four up; doing
// it by hand works until one of them silently 404s at runtime.
export default defineConfig({
  plugins: [alphaTab()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    chunkSizeWarningLimit: 1500,   // alphaTab is legitimately large
  },
  server: {
    port: 5173,
    proxy: {
      // Dev server talks to uvicorn so the app behaves identically in both modes.
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
  resolve: {
    alias: { '@': resolve(import.meta.dirname, 'src') },
  },
});

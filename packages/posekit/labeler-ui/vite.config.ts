import { svelte } from '@sveltejs/vite-plugin-svelte';
import { defineConfig } from 'vitest/config';

// The Python label server (`posekit.apis.label_server`); in dev, Vite serves the UI with HMR and proxies /api to it.
const LABEL_SERVER = process.env.LABEL_SERVER ?? 'http://127.0.0.1:9898';

export default defineConfig({
  plugins: [svelte()],
  // Relative asset URLs: the Python server serves the built directory as it is.
  base: './',
  build: { outDir: '../src/posekit/labeler/static', emptyOutDir: true },
  server: { proxy: { '/api': LABEL_SERVER } },
  test: {
    environment: 'node',
    include: ['src/**/*.spec.ts'],
    expect: { requireAssertions: true }
  }
});

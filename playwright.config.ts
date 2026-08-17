import { defineConfig, devices } from '@playwright/test';

/**
 * A3 (`plan-docs/REMAINING-WORK.md`) — the one end-to-end smoke test.
 *
 * Deliberately separate from `pnpm test` (Vitest, 774 tests, fully offline,
 * no model downloads): this suite drives a real browser through a real
 * chapter, which means a real ~23MB model fetch from the Hugging Face Hub on
 * its first run. Not wired into `predev`/`prebuild` — run explicitly via
 * `pnpm test:e2e`.
 */
export default defineConfig({
  testDir: './tests/e2e',
  timeout: 120_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  retries: 0,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: 'http://localhost:3100',
    trace: 'retain-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: {
    command: 'pnpm exec next dev --port 3100',
    url: 'http://localhost:3100',
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});

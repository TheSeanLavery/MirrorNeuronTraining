import { defineConfig } from "@playwright/test";

const PLAYWRIGHT_PORT = process.env.PW_PORT || process.env.PORT || "4173";
const APP_BASE_URL = process.env.PW_BASE_URL || `http://127.0.0.1:${PLAYWRIGHT_PORT}`;
const PLAYWRIGHT_BROWSER = process.env.PW_BROWSER || "chromium";
const PLAYWRIGHT_EXECUTABLE = process.env.PW_EXECUTABLE;

export default defineConfig({
  testDir: "./tests",
  timeout: 120000,
  expect: {
    timeout: 10000
  },
  use: {
    baseURL: APP_BASE_URL,
    browserName: PLAYWRIGHT_BROWSER,
    headless: false,
    viewport: { width: 1440, height: 900 },
    actionTimeout: 15000,
    ...(PLAYWRIGHT_EXECUTABLE ? { launchOptions: { executablePath: PLAYWRIGHT_EXECUTABLE } } : {}),
  },
  ...(process.env.PW_MANAGE_SERVER === "1"
    ? {
        webServer: {
          command: `npm run dev -- --host 0.0.0.0 --port ${PLAYWRIGHT_PORT}`,
          url: APP_BASE_URL,
          reuseExistingServer: true,
          timeout: 120000
        }
      }
    : {})
});

const { defineConfig } = require('@playwright/test');
const baseURL = process.env.CITEWISE_BASE_URL || 'http://127.0.0.1:8765';

module.exports = defineConfig({
  testDir: './tests/browser',
  reporter: [['list'], ['html', { outputFolder: 'artifacts/phase5/final-browser-report', open: 'never' }], ['json', { outputFile: 'artifacts/phase5/final-browser-report/results.json' }]],
  outputDir: 'artifacts/phase5/final-browser-output',
  use: { baseURL, trace: 'retain-on-failure', screenshot: 'only-on-failure' }
});

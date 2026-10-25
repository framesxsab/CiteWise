const { defineConfig } = require('@playwright/test');
const path = require('node:path');
const baseURL = process.env.CITEWISE_BASE_URL || 'http://127.0.0.1:8765';
const artifactDir = process.env.CITEWISE_ARTIFACT_DIR || 'artifacts/phase5';

module.exports = defineConfig({
  testDir: './tests/browser',
  reporter: [['list'], ['html', { outputFolder: path.join(artifactDir, 'browser-report'), open: 'never' }], ['json', { outputFile: path.join(artifactDir, 'browser-report', 'results.json') }]],
  outputDir: path.join(artifactDir, 'browser-output'),
  use: { baseURL, trace: 'retain-on-failure', screenshot: 'only-on-failure' }
});

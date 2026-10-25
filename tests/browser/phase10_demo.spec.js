const { test, expect } = require('@playwright/test');
const fs = require('node:fs');
const path = require('node:path');

const screenshotDir = process.env.CITEWISE_SCREENSHOT_DIR || 'artifacts/phase10/ui';
const paperId = '1909.09534';
const question = 'Which datasets are used?';
const goldIds = ['1909.09534:s1:p1', '1909.09534:s1:p2'];

test('Phase 10 localhost demo returns real QASPER evidence in the UI', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.goto('/');
  await page.getByLabel('Scientific paper').selectOption(paperId);
  await page.getByLabel('Annotated question (optional)').selectOption({ label: question });
  await page.getByLabel('Retriever').selectOption('e5-base-v2');

  const response = page.waitForResponse(r => r.url().endsWith('/api/search') && r.request().method() === 'POST');
  await page.getByRole('button', { name: 'Retrieve evidence' }).click();
  const apiResponse = await response;
  expect(apiResponse.status()).toBe(200);
  const payload = await apiResponse.json();
  expect(payload.title).toBe('Creative GANs for generating poems, lyrics, and metaphors');
  expect(payload.results).toHaveLength(5);
  expect(payload.results.some(row => goldIds.includes(row.id))).toBe(true);

  const cards = page.locator('article.evidence');
  await expect(cards).toHaveCount(5);
  for (let i = 0; i < payload.results.length; i++) {
    await expect(cards.nth(i)).toContainText(payload.results[i].id);
    await expect(cards.nth(i)).toContainText(payload.results[i].text.slice(0, 80));
  }
  await expect(cards.filter({ hasText: '1909.09534:s1:p1' })).toContainText('14950 metaphor sentences');

  fs.mkdirSync(screenshotDir, { recursive: true });
  await page.screenshot({ path: path.join(screenshotDir, 'demo-e5-datasets.png'), fullPage: true });
  await cards.filter({ hasText: '1909.09534:s1:p1' }).screenshot({
    path: path.join(screenshotDir, 'demo-e5-source-passage.png'),
  });
});

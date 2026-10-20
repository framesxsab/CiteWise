const { test, expect } = require('@playwright/test');
const AxeBuilder = require('@axe-core/playwright').default;
const fs = require('node:fs');

test('real paper evidence matches the API response', async ({ page, request }) => {
  await page.goto('/');
  const papers = await request.get('/api/papers?split=validation').then(r => r.json());
  const paper = papers.find(p => p.paragraphs > 0 && p.questions > 0);
  expect(paper, 'validation must contain an actual paper with paragraphs and questions').toBeTruthy();
  const paperSelect = page.getByLabel('Scientific paper');
  await paperSelect.selectOption(paper.id);
  await expect(paperSelect.locator('option:checked')).toHaveText(paper.title || paper.id);
  await expect(page.getByLabel('Retriever').locator('option[value="bge-base"]')).toContainText('instructed semantic retrieval');
  const questionSelect = page.getByLabel('Annotated question (optional)');
  await expect(questionSelect.locator('option')).toHaveCount(paper.questions + 1);
  const question = await questionSelect.locator('option').nth(1).getAttribute('value');
  await page.getByLabel('Retriever').selectOption('bm25');
  await questionSelect.selectOption({ label: question });
  const expected = await request.post('/api/search', { data: { split: 'validation', paper_id: paper.id, question, model: 'bm25', top_k: 5 } }).then(r => r.json());
  await page.route('**/api/search', async route => {
    await new Promise(resolve => setTimeout(resolve, 150));
    await route.continue();
  });
  const response = page.waitForResponse(r => r.url().endsWith('/api/search') && r.request().method() === 'POST');
  await page.getByRole('button', { name: 'Retrieve evidence' }).click();
  await expect(page.getByRole('status')).toContainText('Retrieving evidence');
  await response;
  await expect(page.locator('article.evidence')).toHaveCount(expected.results.length);
  await expect(page.locator('article.evidence').first()).toContainText(expected.results[0].text);
  await expect(page.locator('article.evidence').first()).toContainText(expected.results[0].id);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(1280);
  fs.mkdirSync('artifacts/phase5/ui/final', { recursive: true });
  await page.screenshot({ path: 'artifacts/phase5/ui/final/evidence-desktop.png', fullPage: true });
});

test('BGE-base selection runs the cached neural retriever against real paper text', async ({ page, request }) => {
  test.setTimeout(120000);
  await page.goto('/');
  const papers = await request.get('/api/papers?split=validation').then(r => r.json());
  const paper = papers.filter(p => p.paragraphs >= 5 && p.questions > 0).sort((a, b) => a.paragraphs - b.paragraphs)[0];
  await page.getByLabel('Scientific paper').selectOption(paper.id);
  const questionSelect = page.getByLabel('Annotated question (optional)');
  const question = await questionSelect.locator('option').nth(1).getAttribute('value');
  await page.getByLabel('Retriever').selectOption('bge-base');
  await questionSelect.selectOption({ label: question });
  const expected = await request.post('/api/search', { timeout: 120000, data: { split: 'validation', paper_id: paper.id, question, model: 'bge-base', top_k: 5 } }).then(r => r.json());
  expect(expected.results).toHaveLength(5);
  await page.getByRole('button', { name: 'Retrieve evidence' }).click();
  await expect(page.locator('article.evidence')).toHaveCount(5, { timeout: 120000 });
  await expect(page.locator('article.evidence').first()).toContainText(expected.results[0].id);
  await expect(page.locator('article.evidence').first()).toContainText(expected.results[0].text);
});

test('short questions and malformed API inputs fail clearly; keyboard focus is visible', async ({ page, request }) => {
  await page.goto('/');
  await page.getByLabel('Scientific paper').selectOption({ index: 1 });
  await expect(page.getByRole('button', { name: 'Retrieve evidence' })).toBeEnabled();
  await page.getByRole('textbox', { name: 'Question' }).fill('x');
  await page.getByRole('button', { name: 'Retrieve evidence' }).click();
  await expect(page.getByRole('status')).toContainText('at least three characters');
  await page.getByLabel('Dataset split').focus();
  await page.keyboard.press('Tab');
  await expect(page.getByLabel('Retriever')).toBeFocused();
  expect(await page.getByLabel('Retriever').evaluate(el => getComputedStyle(el).outlineStyle)).not.toBe('none');
  const invalid = await request.post('/api/search', { data: { split: 'invalid', paper_id: 'missing', question: 'ab' } });
  expect(invalid.status()).toBe(422);
});

for (const width of [390, 375]) {
  test(`layout fits ${width}px mobile width`, async ({ page }) => {
    await page.setViewportSize({ width, height: 844 });
    await page.goto('/');
    await expect(page.getByRole('heading', { name: 'CiteWise' })).toBeVisible();
    await page.getByLabel('Scientific paper').selectOption({ index: 1 });
    const questions = page.getByLabel('Annotated question (optional)');
    await questions.selectOption({ index: 1 });
    await page.getByLabel('Retriever').selectOption('bm25');
    await page.getByRole('button', { name: 'Retrieve evidence' }).click();
    await expect(page.locator('article.evidence')).toHaveCount(5);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
    fs.mkdirSync('artifacts/phase5/ui/final', { recursive: true });
    await page.screenshot({ path: `artifacts/phase5/ui/final/mobile-${width}.png`, fullPage: true });
    if (width === 390) {
      await page.locator('article.evidence').first().scrollIntoViewIfNeeded();
      await page.screenshot({ path: 'artifacts/phase5/ui/final/mobile-390-evidence.png' });
    }
  });
}

test('axe accessibility scan records current violations', async ({ page }, testInfo) => {
  await page.goto('/');
  await expect(page.getByRole('status')).toContainText('real papers loaded');
  const result = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']).analyze();
  await fs.promises.mkdir('artifacts/phase5/a11y/final', { recursive: true });
  await fs.promises.writeFile('artifacts/phase5/a11y/final/axe-results.json', JSON.stringify({ url: page.url(), violations: result.violations }, null, 2));
  await testInfo.attach('axe-violations', { body: Buffer.from(JSON.stringify(result.violations, null, 2)), contentType: 'application/json' });
  expect(result.violations.filter(v => ['critical', 'serious'].includes(v.impact)), 'critical and serious axe findings').toEqual([]);
});

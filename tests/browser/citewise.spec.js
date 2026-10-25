const { test, expect } = require('@playwright/test');
const AxeBuilder = require('@axe-core/playwright').default;
const fs = require('node:fs');
const path = require('node:path');
const screenshotDir = process.env.CITEWISE_SCREENSHOT_DIR || 'artifacts/phase5/ui/final';
const a11yDir = process.env.CITEWISE_A11Y_DIR || 'artifacts/phase5/a11y/final';

test('real paper evidence matches the API response', async ({ page, request }) => {
  await page.goto('/');
  const papers = await request.get('/api/papers?split=validation').then(r => r.json());
  const paper = papers.find(p => p.paragraphs > 0 && p.questions > 0);
  expect(paper, 'validation must contain an actual paper with paragraphs and questions').toBeTruthy();
  const paperSelect = page.getByLabel('Scientific paper');
  await paperSelect.selectOption(paper.id);
  await expect(paperSelect.locator('option:checked')).toHaveText(paper.title || paper.id);
  await expect(page.getByLabel('Retriever').locator('option[value="bge-base"]')).toContainText('instructed neural retriever');
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
  await expect(page.locator('#status')).toContainText('Retrieving evidence');
  await response;
  await expect(page.locator('article.evidence')).toHaveCount(expected.results.length);
  await expect(page.locator('article.evidence').first()).toContainText(expected.results[0].text);
  await expect(page.locator('article.evidence').first()).toContainText(expected.results[0].id);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(1280);
  fs.mkdirSync(screenshotDir, { recursive: true });
  await page.screenshot({ path: path.join(screenshotDir, 'evidence-desktop.png'), fullPage: true });
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

test('comparison page uses saved six-model validation results and pinned model settings', async ({ page, request }) => {
  await page.goto('/');
  const saved = await request.get('/api/research').then(r => r.json());
  expect(saved.evaluation.split).toBe('validation');
  expect(saved.evaluation.question_count).toBe(888);
  expect(saved.evaluation.test_split_accessed).toBe(false);
  expect(saved.evaluation.models).toHaveLength(6);
  expect(saved.evaluation.models.every(m => m.status === 'evaluated')).toBe(true);
  expect(saved.evaluation.paired_comparisons).toHaveLength(45);
  expect(saved.evaluation.neural_recall_at_1_comparisons).toHaveLength(6);
  const config = await request.get('/api/config').then(r => r.json());
  expect(config.configured_default_model).toBe('e5-base-v2');
  expect(config.default_model).toBe(config.available_models.includes('e5-base-v2') ? 'e5-base-v2' : 'bm25');
  await expect(page.getByRole('heading', { name: 'Model comparison — QASPER validation' })).toBeVisible();
  await expect(page.getByRole('img', { name: 'Recall@5 on QASPER validation' })).toBeVisible();
  await expect(page.getByRole('cell', { name: 'MiniLM-L6-v2', exact: false })).toBeVisible();
  await expect(page.getByRole('cell', { name: 'E5-base-v2', exact: false })).toBeVisible();
  await expect(page.locator('#comparison-table')).toContainText('54.67%');
  await expect(page.locator('#comparison-table')).toContainText('45.69%');
  await expect(page.locator('#comparison-table')).toContainText('888 / 276');
  await expect(page.locator('#chart-quality-latency')).toHaveAttribute('role', 'img');
  await expect(page.getByText('Checkpoint provenance: the resumed lexical baseline checkpoints retain the original source fingerprint', { exact: false })).toBeVisible();
  await page.getByText('Neural checkpoint revisions and encoding settings').click();
  await expect(page.getByRole('cell', { name: /query:/ })).toBeVisible();
  await expect(page.getByRole('cell', { name: '1110a243fdf4706b3f48f1d95db1a4f5529b4d41' })).toBeVisible();
  await expect(page.getByRole('cell', { name: '1c644c92ad3ba1efdad3f1451a637716616a20e8' })).toBeVisible();
  await expect(page.locator('#checkpoint-table')).toContainText('9c213a7880437305ed6fc0f81ff244b6e6f78963ac24c7030f25a845b2c543db');
  await page.getByText('Paired neural model comparisons').click();
  await expect(page.locator('#paired-table tbody tr')).toHaveCount(12);
  expect((await request.get('/api/papers?split=test')).status()).toBe(422);
  fs.mkdirSync(screenshotDir, { recursive: true });
  await page.screenshot({ path: path.join(screenshotDir, 'comparison-desktop.png'), fullPage: true });
  await page.setViewportSize({ width: 1280, height: 1056 });
  await page.locator('#comparison-title').evaluate(node => node.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: path.join(screenshotDir, 'comparison-dashboard.png') });
});

for (const model of ['minilm', 'e5-base-v2', 'gte-base']) {
  test(`${model} selection runs real validation inference and returns source evidence`, async ({ page, request }) => {
    test.setTimeout(120000);
    const config = await request.get('/api/config').then(r => r.json());
    expect(config.available_models).toContain(model);
    await page.goto('/');
    const papers = await request.get('/api/papers?split=validation').then(r => r.json());
    const paper = papers.filter(p => p.paragraphs >= 5 && p.questions > 0).sort((a,b) => a.paragraphs-b.paragraphs)[0];
    await page.getByLabel('Scientific paper').selectOption(paper.id);
    const questions = page.getByLabel('Annotated question (optional)');
    const question = await questions.locator('option').nth(1).getAttribute('value');
    await page.getByLabel('Retriever').selectOption(model);
    await questions.selectOption({ label: question });
    const expected = await request.post('/api/search', { timeout: 120000, data: {
      split: 'validation', paper_id: paper.id, question, model, top_k: 5,
    }}).then(r => r.json());
    expect(expected.results).toHaveLength(5);
    await page.getByRole('button', { name: 'Retrieve evidence' }).click();
    await expect(page.locator('article.evidence')).toHaveCount(5, { timeout: 120000 });
    await expect(page.locator('article.evidence').first()).toContainText(expected.results[0].id);
    await expect(page.locator('article.evidence').first()).toContainText(expected.results[0].text);
  });
}

test('short questions and malformed API inputs fail clearly; keyboard focus is visible', async ({ page, request }) => {
  await page.goto('/');
  await page.getByLabel('Scientific paper').selectOption({ index: 1 });
  await expect(page.getByRole('button', { name: 'Retrieve evidence' })).toBeEnabled();
  await page.getByRole('textbox', { name: 'Question' }).fill('x');
  await page.getByRole('button', { name: 'Retrieve evidence' }).click();
  await expect(page.locator('#status')).toContainText('at least three characters');
  await page.getByLabel('Retriever').focus();
  await expect(page.getByLabel('Retriever')).toBeFocused();
  expect(await page.getByLabel('Retriever').evaluate(el => getComputedStyle(el).outlineStyle)).not.toBe('none');
  await page.keyboard.press('Tab');
  await expect(page.getByLabel('Scientific paper')).toBeFocused();
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
    fs.mkdirSync(screenshotDir, { recursive: true });
    await page.screenshot({ path: path.join(screenshotDir, `mobile-${width}.png`), fullPage: true });
    if (width === 390) {
      await page.locator('article.evidence').first().scrollIntoViewIfNeeded();
      await page.screenshot({ path: path.join(screenshotDir, 'mobile-390-evidence.png') });
      await page.locator('article.evidence').first().screenshot({
        path: path.join(screenshotDir, 'mobile-390-evidence-card.png'),
      });
    }
  });
}

test('axe accessibility scan records current violations', async ({ page }, testInfo) => {
  await page.goto('/');
  await expect(page.locator('#status')).toContainText('real papers loaded');
  const result = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']).analyze();
  await fs.promises.mkdir(a11yDir, { recursive: true });
  await fs.promises.writeFile(path.join(a11yDir, 'axe-results.json'), JSON.stringify({ url: page.url(), violations: result.violations }, null, 2));
  await testInfo.attach('axe-violations', { body: Buffer.from(JSON.stringify(result.violations, null, 2)), contentType: 'application/json' });
  expect(result.violations.filter(v => ['critical', 'serious'].includes(v.impact)), 'critical and serious axe findings').toEqual([]);
});

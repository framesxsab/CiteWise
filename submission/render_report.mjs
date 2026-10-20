import { chromium } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const dir = path.dirname(fileURLToPath(import.meta.url));
const pdf = path.join(dir, 'CiteWise_Final_Report.pdf');
const qa = path.join(dir, 'qa');
fs.mkdirSync(qa, { recursive: true });
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 816, height: 1056 }, deviceScaleFactor: 1 });
  await page.goto(`file://${path.join(dir, 'CiteWise_Final_Report.html').replaceAll('\\', '/')}`);
  const pages = page.locator('.page');
  const count = await pages.count();
  if (count !== 10) throw new Error(`Expected 10 pages, found ${count}`);
  const overflow = await pages.evaluateAll(ns => ns.map((n, i) => ({ i: i + 1, h: n.scrollHeight, ch: n.clientHeight, w: n.scrollWidth, cw: n.clientWidth })).filter(x => x.h > x.ch + 1 || x.w > x.cw + 1));
  if (overflow.length) throw new Error(`Content overflow: ${JSON.stringify(overflow)}`);
  for (let i = 0; i < count; i++) await pages.nth(i).screenshot({ path: path.join(qa, `page-${String(i + 1).padStart(2, '0')}.png`) });
  await page.pdf({ path: pdf, format: 'Letter', printBackground: true, preferCSSPageSize: true, margin: { top: 0, right: 0, bottom: 0, left: 0 } });
  const pdfPages = (fs.readFileSync(pdf).toString('latin1').match(/\/Type\s*\/Page\b/g) || []).length;
  if (pdfPages !== 10) throw new Error(`Expected 10 PDF pages, found ${pdfPages}`);
  console.log(`PASS: ${pdfPages}-page report; no page overflow; screenshots: ${qa}`);
} finally { await browser.close(); }

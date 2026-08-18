/**
 * Drive the real UI in a real browser and photograph it.
 *
 * This exists because layout cannot be verified by reading CSS. Two numbers sat
 * hardcoded in this app for weeks -- a 19rem sidebar and a 42% notation split --
 * and no amount of care while writing them would have shown how they behaved on
 * a laptop screen. Looking at it does.
 *
 *     node test/screenshot.mjs [baseUrl] [outDir]
 *
 * It also fails on console errors, which is the cheapest possible check that a
 * page actually runs rather than merely builds.
 */

import { chromium } from 'playwright';
import { mkdirSync, existsSync } from 'node:fs';

const BASE = process.argv[2] || 'http://127.0.0.1:8011';
const OUT = process.argv[3] || 'screenshots';

// The shapes that actually matter: a desktop, the laptop most work happens on,
// a tablet at the breakpoint, and a phone.
const VIEWPORTS = [
  { name: 'desktop', width: 1680, height: 1050 },
  { name: 'laptop', width: 1280, height: 800 },
  { name: 'tablet', width: 860, height: 1000 },
  { name: 'phone', width: 480, height: 900 },
];

mkdirSync(OUT, { recursive: true });

// Use the Chromium already on the machine rather than downloading one to match
// this package's pinned revision.
const PREINSTALLED = [
  '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
  '/opt/pw-browsers/chromium/chrome-linux/chrome',
];
const executablePath = process.env.CHROMIUM_PATH || PREINSTALLED.find(existsSync);

const browser = await chromium.launch(executablePath ? { executablePath } : {});
const errors = [];
let failures = 0;

function check(name, condition, detail = '') {
  if (condition) {
    console.log(`  ok   ${name}`);
  } else {
    failures += 1;
    console.log(`  FAIL ${name}${detail ? ` -- ${detail}` : ''}`);
  }
}

for (const viewport of VIEWPORTS) {
  const context = await browser.newContext({
    viewport: { width: viewport.width, height: viewport.height },
    deviceScaleFactor: 1,
  });
  const page = await context.newPage();

  page.on('console', (msg) => {
    if (msg.type() === 'error') errors.push(`[${viewport.name}] ${msg.text()}`);
  });
  page.on('pageerror', (err) => errors.push(`[${viewport.name}] ${err.message}`));

  console.log(`\n${viewport.name} (${viewport.width}x${viewport.height})`);

  await page.goto(BASE, { waitUntil: 'networkidle' });

  // Open the seeded song so the screenshot shows a populated app rather than
  // an empty state that hides every layout problem. On a narrow screen the
  // library is off-canvas, which is the intended behaviour, so it has to be
  // summoned the same way a person would.
  const narrowLayout = viewport.width <= 900;
  if (narrowLayout) {
    await page.locator('#sidebartoggle').click();
    await page.waitForTimeout(300);
  }

  const song = page.locator('.songitem').first();
  if (await song.count()) {
    await song.click();
    await page.waitForTimeout(2500);      // notation render
  }

  // Picking a song on a narrow screen should hand the screen back rather than
  // leaving the library covering the score.
  if (narrowLayout) {
    const stillOpen = await page.evaluate(
      () => document.body.classList.contains('sidebar-open')
    );
    check('the library closes itself after picking a song', !stillOpen);
  }

  await page.screenshot({ path: `${OUT}/${viewport.name}.png`, fullPage: false });

  // The page must never scroll sideways. This is the single most common
  // responsive failure and it is invisible until someone hits it.
  const overflows = await page.evaluate(
    () => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1
  );
  check('no horizontal overflow', !overflows);

  // Below the breakpoint the sidebar must get out of the way rather than
  // squeezing the score into a column.
  const narrow = narrowLayout;
  const sidebarVisible = await page.evaluate(() => {
    const el = document.querySelector('#sidebar');
    if (!el) return false;
    const box = el.getBoundingClientRect();
    return box.right > 8;      // still on screen
  });
  check(narrow ? 'sidebar is out of the way' : 'sidebar is visible',
    narrow ? !sidebarVisible : sidebarVisible);

  // The transport is useless if it is off screen.
  for (const id of ['#playpause', '#source']) {
    const onScreen = await page.evaluate((sel) => {
      const el = document.querySelector(sel);
      if (!el) return false;
      const box = el.getBoundingClientRect();
      return box.width > 0 && box.right <= window.innerWidth + 1 && box.top >= 0;
    }, id);
    check(`${id} is reachable`, onScreen);
  }

  // Both panes need real height, or the splitter cannot be grabbed to fix it.
  const panes = await page.evaluate(() => {
    const n = document.querySelector('.notation-wrap')?.getBoundingClientRect().height ?? 0;
    const g = document.querySelector('.editor-wrap')?.getBoundingClientRect().height ?? 0;
    return { notation: Math.round(n), grid: Math.round(g) };
  });
  check('notation pane has height', panes.notation > 60, `${panes.notation}px`);
  check('grid pane has height', panes.grid > 60, `${panes.grid}px`);

  await context.close();
}

// --- the splitter, which is the whole point of "adjustable" -----------------

console.log('\nsplitter');
{
  const context = await browser.newContext({ viewport: { width: 1280, height: 800 } });
  const page = await context.newPage();
  page.on('pageerror', (err) => errors.push(`[splitter] ${err.message}`));
  await page.goto(BASE, { waitUntil: 'networkidle' });
  await page.locator('.songitem').first().click();
  await page.waitForTimeout(2000);

  const before = await page.evaluate(
    () => document.querySelector('.notation-wrap').getBoundingClientRect().height
  );

  // Drag to an explicit target rather than by a pixel offset. Any fixed
  // distance is worth a different share once a row is added above the panes,
  // which is what turned this check into a false failure.
  const geometry = await page.evaluate(() => {
    const top = document.querySelector('.notation-wrap').getBoundingClientRect().top;
    const box = document.querySelector('.main').getBoundingClientRect();
    return { top, available: box.bottom - top, height: box.height };
  });
  // Aim for a pane a quarter of the container tall, which is what dropping the
  // handle there should produce.
  const wantedHeight = geometry.height * 0.25;
  const targetY = geometry.top + wantedHeight;

  const handle = page.locator('#splitter');
  const box = await handle.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2, targetY, { steps: 15 });
  await page.mouse.up();
  await page.waitForTimeout(300);

  const after = await page.evaluate(
    () => document.querySelector('.notation-wrap').getBoundingClientRect().height
  );
  check('the splitter drags the notation pane to where it was dropped',
    Math.abs(after - wantedHeight) < 25,
    `${Math.round(before)}px -> ${Math.round(after)}px, wanted ~${Math.round(wantedHeight)}px`);

  // A layout you must re-establish every visit is not adjustable in any useful
  // sense, so the split has to survive a reload.
  await page.reload({ waitUntil: 'networkidle' });
  await page.waitForTimeout(1200);
  const restored = await page.evaluate(
    () => document.querySelector('.notation-wrap').getBoundingClientRect().height
  );
  check('the split survives a reload',
    Math.abs(restored - after) < 40, `${Math.round(after)}px -> ${Math.round(restored)}px`);

  await page.screenshot({ path: `${OUT}/splitter-dragged.png` });
  await context.close();
}

// --- playing a part in ------------------------------------------------------

console.log('\nrecording');
{
  const context = await browser.newContext({ viewport: { width: 1280, height: 800 } });
  const page = await context.newPage();
  page.on('pageerror', (err) => errors.push(`[recording] ${err.message}`));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(`[recording] ${m.text()}`); });

  await page.goto(BASE, { waitUntil: 'networkidle' });
  await page.locator('.songitem').first().click();
  await page.waitForTimeout(2500);

  await page.locator('#opensettings').click();
  await page.waitForTimeout(400);

  const lanes = await page.locator('.binding').count();
  check('every lane is bindable', lanes >= 9, `${lanes} listed`);

  const kickIsSpace = await page.evaluate(() =>
    [...document.querySelectorAll('.binding')]
      .find((row) => row.textContent.includes('Kick'))?.textContent.includes('Space'));
  check('the kick is on the space bar', !!kickIsSpace);

  // No count-in, so the check does not sit through a bar of clicks.
  await page.selectOption('#countin', '0');
  await page.keyboard.press('Escape');
  await page.waitForTimeout(300);

  const slug = await page.evaluate(() => new URLSearchParams(location.search).get('song'));

  // Clear the opening bars first. Without this the check passes or fails
  // depending on whether a previous run already wrote the same notes to the
  // same slots, which made it report a false failure the second time.
  const noteCount = () => page.evaluate(async (s) => {
    const r = await fetch(`/api/songs/${s}/chart`);
    const c = await r.json();
    return c.bars.slice(0, 4)
      .flatMap((b) => Object.values(b.lanes))
      .join('').replace(/-/g, '').length;
  }, slug);

  await page.evaluate(async (s) => {
    const r = await fetch(`/api/songs/${s}/chart`);
    const chart = await r.json();
    for (const bar of chart.bars.slice(0, 4)) bar.lanes = {};
    await fetch(`/api/songs/${s}/chart`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(chart),
    });
  }, slug);
  await page.reload({ waitUntil: 'networkidle' });
  await page.waitForTimeout(2500);

  check('the opening bars start empty', (await noteCount()) === 0);

  await page.locator('#record').click();
  await page.waitForTimeout(600);
  check('record arms', await page.evaluate(
    () => document.querySelector('#record').classList.contains('armed')));

  for (const key of ['Space', 'KeyF', 'KeyD', 'KeyJ', 'Space', 'KeyK']) {
    await page.keyboard.press(key);
    await page.waitForTimeout(140);
  }
  await page.waitForTimeout(1500);

  // Headless has no audio device, so alphaTab never reports a position and the
  // recorder correctly refuses to place hits. Which of the two behaviours to
  // assert depends on whether playback actually started.
  const playing = await page.evaluate(() => {
    const el = document.querySelector('#status')?.textContent || '';
    return !el.includes('nothing is playing');
  });

  if (playing) {
    const written = await noteCount();
    check('played keys reach the chart', written > 0, `${written} notes written`);
  } else {
    const told = await page.evaluate(
      () => (document.querySelector('#status')?.textContent || '').toLowerCase());
    check('a refused hit says why rather than vanishing',
      told.includes('nothing is playing') || told.includes('press play'), told.slice(0, 70));
    check('nothing was written while not playing', (await noteCount()) === 0);
  }

  await page.screenshot({ path: `${OUT}/recording.png` });

  await page.locator('#record').click();
  await page.waitForTimeout(300);
  check('record disarms', !(await page.evaluate(
    () => document.querySelector('#record').classList.contains('armed'))));

  await context.close();
}

await browser.close();

if (errors.length) {
  console.log('\nconsole errors:');
  for (const line of [...new Set(errors)]) console.log(`  ${line}`);
  failures += errors.length;
}

console.log(failures ? `\n${failures} check(s) failed` : '\nall UI checks passed');
console.log(`screenshots in ${OUT}/`);
process.exit(failures ? 1 : 0);

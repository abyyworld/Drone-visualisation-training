/**
 * End-to-end browser test for the inspection web app.
 *
 * Runs the real pipeline — ONNX Runtime Web, letterboxing, YOLO decoding, NMS, severity
 * scoring, canvas rendering — against the hand-computed fixtures from make_fixtures.py, so
 * the expected boxes and scores below are arithmetic, not snapshots.
 *
 *   python3 tests/make_fixtures.py
 *   node tests/test_web.mjs
 *
 * onnxruntime-web is served from a local copy rather than the CDN (see `runtime.ortBase` in
 * the generated manifest), so the suite runs offline and does not depend on jsdelivr.
 */

import { createServer } from 'node:http';
import { createRequire } from 'node:module';
import { readFile, cp, mkdir, writeFile, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { extname, join, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { tmpdir } from 'node:os';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const FIXTURES = join(ROOT, 'tests', 'fixtures');
const SITE = join(tmpdir(), 'drone-inspection-test-site');

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.css': 'text/css', '.json': 'application/json', '.onnx': 'application/octet-stream',
  '.wasm': 'application/wasm', '.png': 'image/png', '.map': 'application/json',
};

// --- expectations, derived from the fixtures -------------------------------------------
// turbine: corrosion 0.90 (weight 2.0) + crack 0.80 (weight 3.0) = 1.80 + 2.40 = 4.20
// solar:   soiling 0.85 (weight 1.0) + missing_module 0.75 (weight 4.0) = 0.85 + 3.00 = 3.85
// Both land in [2, 5) -> "Moderate damage".
const EXPECT = {
  turbine: { score: '4.20', severity: 'Moderate damage', detections: ['corrosion — 90.0%', 'crack — 80.0%'] },
  solar: { score: '3.85', severity: 'Moderate damage', detections: ['soiling — 85.0%', 'missing_module — 75.0%'] },
};

let failures = 0;

function check(name, condition, detail = '') {
  if (condition) {
    console.log(`  PASS  ${name}`);
  } else {
    failures += 1;
    console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
  }
}

function equal(name, actual, expected) {
  check(name, actual === expected, `expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
}

async function buildSite() {
  await rm(SITE, { recursive: true, force: true });
  await cp(join(ROOT, 'web'), SITE, { recursive: true });

  // Fixture models stand in for trained ones.
  for (const model of ['gate.onnx', 'turbine.onnx', 'solar.onnx']) {
    await cp(join(FIXTURES, model), join(SITE, 'models', model));
  }

  // Serve onnxruntime-web locally so the suite does not need network access.
  const ortDist = join(ROOT, 'node_modules', 'onnxruntime-web', 'dist');
  const ortSource = existsSync(ortDist) ? ortDist : process.env.ORT_DIST;
  if (!ortSource || !existsSync(ortSource)) {
    throw new Error(
      'onnxruntime-web not found. Run `npm install onnxruntime-web@1.23.0` in the repo root, '
      + 'or set ORT_DIST to an existing dist/ directory.',
    );
  }
  await mkdir(join(SITE, 'ort'), { recursive: true });
  await cp(ortSource, join(SITE, 'ort'), { recursive: true });

  const manifest = JSON.parse(await readFile(join(ROOT, 'web', 'models', 'manifest.json'), 'utf8'));
  manifest.runtime = { ortBase: '/ort/' };
  await writeFile(join(SITE, 'models', 'manifest.json'), JSON.stringify(manifest, null, 2));
}

function serve() {
  const server = createServer(async (req, res) => {
    const path = join(SITE, decodeURIComponent(req.url.split('?')[0]));
    const file = path.endsWith('/') ? join(path, 'index.html') : path;
    try {
      const body = await readFile(file);
      res.writeHead(200, {
        'content-type': MIME[extname(file)] ?? 'application/octet-stream',
        'content-length': body.length,
      });
      res.end(req.method === 'HEAD' ? undefined : body);
    } catch {
      res.writeHead(404).end('not found');
    }
  });
  return new Promise((done) => server.listen(0, () => done({ server, port: server.address().port })));
}

async function cardFor(page, filename) {
  return page.locator('.card').filter({ has: page.locator('.card__title', { hasText: filename }) });
}

async function main() {
  if (!existsSync(join(FIXTURES, 'gate.onnx'))) {
    throw new Error('fixtures missing — run `python3 tests/make_fixtures.py` first');
  }

  await buildSite();
  const { server, port } = await serve();
  const browser = await chromium.launch();
  const page = await browser.newPage();

  const consoleErrors = [];
  page.on('pageerror', (error) => consoleErrors.push(String(error)));

  try {
    await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'networkidle' });

    console.log('\nPage load');
    equal('title', await page.title(), 'Drone Inspection — Turbine & Solar Defect Analysis');
    check('backend reported', /WebGPU|WASM/.test(await page.locator('#backend').textContent()));
    check('no model-missing banner', await page.locator('#status-banner').isHidden());
    check('domain override stays hidden when gate is present',
      await page.locator('#override-row').isHidden());

    console.log('\nTurbine image (red)');
    await page.locator('#file-input').setInputFiles(join(FIXTURES, 'turbine_red.png'));
    await page.waitForSelector('.card', { timeout: 60000 });
    const turbine = await cardFor(page, 'turbine_red.png');
    await turbine.locator('.badge').waitFor({ timeout: 30000 });

    const turbineBadge = await turbine.locator('.badge').textContent();
    equal('severity badge', turbineBadge, `${EXPECT.turbine.severity} · score ${EXPECT.turbine.score}`);
    const turbineDetections = await turbine.locator('.detections li').allTextContents();
    equal('detection count after NMS (3 raw boxes -> 2)', turbineDetections.length, 2);
    equal('detections', turbineDetections.map((t) => t.trim()).sort().join(' | '),
      EXPECT.turbine.detections.slice().sort().join(' | '));
    check('routed to turbine detector',
      (await turbine.locator('.card__meta').first().textContent()).includes('Wind turbine blade'));
    equal('annotated canvas at native resolution',
      await turbine.locator('canvas').getAttribute('width'), '960');

    console.log('\nSolar image (blue)');
    await page.locator('#file-input').setInputFiles(join(FIXTURES, 'solar_blue.png'));
    const solar = await cardFor(page, 'solar_blue.png');
    await solar.locator('.badge').waitFor({ timeout: 30000 });
    equal('severity badge', await solar.locator('.badge').textContent(),
      `${EXPECT.solar.severity} · score ${EXPECT.solar.score}`);
    equal('detections', (await solar.locator('.detections li').allTextContents())
      .map((t) => t.trim()).sort().join(' | '), EXPECT.solar.detections.slice().sort().join(' | '));
    check('RGB-only limitation is stated on the card',
      (await solar.locator('.card__note').textContent()).includes('thermal infrared'));

    console.log('\nInvalid image (green) — the rejection path');
    await page.locator('#file-input').setInputFiles(join(FIXTURES, 'invalid_green.png'));
    const invalid = await cardFor(page, 'invalid_green.png');
    await invalid.locator('.card__message').waitFor({ timeout: 30000 });
    equal('rejection message', (await invalid.locator('.card__message').textContent()).trim(),
      'Please upload a valid turbine blade or solar panel image.');
    check('card marked rejected', (await invalid.getAttribute('class')).includes('card--rejected'));
    check('no detections drawn on a rejected image',
      await invalid.locator('.detections').count() === 0);

    console.log('\nNon-image file');
    await page.locator('#file-input').setInputFiles(join(FIXTURES, 'not_an_image.txt'));
    await page.waitForTimeout(300);
    equal('banner', (await page.locator('#status-banner').textContent()).trim(),
      'Skipped 1 non-image file.');
    equal('no card created for it', await page.locator('.card').count(), 3);

    console.log('\nSummary and export');
    const tiles = await page.locator('.summary__tiles .tile').allTextContents();
    check('two moderate results counted', /2Moderate/.test(tiles.join('')), tiles.join(' / '));
    check('one rejection counted', /1Rejected/.test(tiles.join('')), tiles.join(' / '));
    equal('overall verdict', (await page.locator('.summary__overall').textContent()).trim(),
      'Moderate damage detected');
    for (const id of ['export-json', 'export-images', 'print-report', 'clear']) {
      check(`${id} enabled`, await page.locator(`#${id}`).isEnabled());
    }

    const download = await Promise.all([
      page.waitForEvent('download'),
      page.locator('#export-json').click(),
    ]).then(([d]) => d);
    const exported = JSON.parse(await readFile(await download.path(), 'utf8'));
    equal('JSON export result count', exported.results.length, 3);
    equal('JSON export overall', exported.summary.overall, 'Moderate damage detected');
    equal('JSON export bbox is in original pixels',
      JSON.stringify(exported.results[0].detections[0].bbox), '[380,430,580,530]');

    await page.locator('.card').first().scrollIntoViewIfNeeded();
    await page.screenshot({ path: join(ROOT, 'docs', 'web-app.png') });

    console.log('\nClear');
    await page.locator('#clear').click();
    equal('cards removed', await page.locator('.card').count(), 0);
    check('export disabled again', await page.locator('#export-json').isDisabled());

    console.log('\nRuntime');
    check('no uncaught page errors', consoleErrors.length === 0, consoleErrors.join('\n          '));

    // Degraded mode: the site must stay honest when weights have not been deployed yet,
    // which is exactly the state a fresh clone is in.
    console.log('\nDegraded mode — no models deployed');
    for (const model of ['gate.onnx', 'turbine.onnx', 'solar.onnx']) {
      await rm(join(SITE, 'models', model), { force: true });
    }
    await page.goto(`http://127.0.0.1:${port}/?nomodels`, { waitUntil: 'networkidle' });
    const banner = (await page.locator('#status-banner').textContent()).trim();
    check('explains that no models are deployed', banner.includes('No detection models are deployed yet'),
      banner);
    check('names the deployment path', banner.includes('tools/export_onnx.py'), banner);
    equal('upload disabled', await page.locator('#drop').getAttribute('aria-disabled'), 'true');
    check('page still renders rather than erroring', await page.locator('.empty').isVisible());

    // Gate absent but a detector present: fall back to a manual choice instead of guessing.
    console.log('\nDegraded mode — detector without gate');
    await cp(join(FIXTURES, 'turbine.onnx'), join(SITE, 'models', 'turbine.onnx'));
    await page.goto(`http://127.0.0.1:${port}/?nogate`, { waitUntil: 'networkidle' });
    check('offers manual inspection type', await page.locator('#override-row').isVisible());
    equal('preselects the only available detector',
      await page.locator('#domain-override').inputValue(), 'turbine');
    check('warns that rejection is unavailable',
      (await page.locator('#status-banner').textContent()).includes('cannot be rejected'));
  } finally {
    await browser.close();
    server.close();
  }

  console.log(failures ? `\n${failures} check(s) failed.` : '\nAll checks passed.');
  return failures ? 1 : 0;
}

main().then((code) => process.exit(code), (error) => {
  console.error(error);
  process.exit(1);
});

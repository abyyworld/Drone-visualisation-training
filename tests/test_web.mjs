/**
 * End-to-end browser test for the inspection web app.
 *
 * Runs the real pipeline - ONNX Runtime Web, letterboxing, YOLO decoding, NMS, severity
 * scoring, canvas rendering - against the hand-computed fixtures from make_fixtures.py, so
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
// crowd:   dense_packing 0.85 (weight 2.0) + choke_point 0.75 (weight 3.0) = 1.70 + 2.25 = 3.95
// wildfire: smoke 0.80 (weight 1.5) + person 0.60 (weight 5.0) = 1.20 + 3.00 = 4.20
// All four land in the middle band. Its name is per subject: a blade takes moderate damage,
// a crowd comes under pressure, a fire is simply active. See LABELS in web/js/severity.js.
const EXPECT = {
  turbine: { score: '4.20', severity: 'Moderate damage', detections: ['corrosion - 90.0%', 'crack - 80.0%'] },
  solar: { score: '3.85', severity: 'Moderate damage', detections: ['soiling - 85.0%', 'missing_module - 75.0%'] },
  crowd: { score: '3.95', severity: 'Under pressure', detections: ['dense_packing - 85.0%', 'choke_point - 75.0%'] },
  wildfire: { score: '4.20', severity: 'Active', detections: ['smoke - 80.0%', 'person - 60.0%'] },
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
  for (const model of ['gate.onnx', 'turbine.onnx', 'solar.onnx', 'crowd.onnx', 'wildfire.onnx']) {
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

  // The turbine entry in the shipped manifest is deliberately empty: it is waiting on a
  // model, and its labels and weights described a dataset that has since been removed. The
  // fixture .onnx has its own fixed three classes, so the test declares them here rather
  // than reading them out of a file that legitimately changes with every deployment. What
  // is under test is the decode-and-score arithmetic, not the manifest's current contents.
  manifest.turbine = {
    ...manifest.turbine,
    labels: ['corrosion', 'crack', 'surface_peeling'],
    severityWeights: { corrosion: 2.0, crack: 3.0, surface_peeling: 1.5 },
  };

  // Same reasoning for crowd: its shipped labels are empty until a model exists, and the
  // fixture has its own four. The manifest's crowd weights are real and are used as they
  // are, because the arithmetic above depends on them.
  manifest.crowd = {
    ...manifest.crowd,
    labels: ['dense_packing', 'counterflow', 'choke_point', 'person_down'],
  };

  manifest.wildfire = {
    ...manifest.wildfire,
    labels: ['fire', 'smoke', 'person'],
  };
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
    throw new Error('fixtures missing - run `python3 tests/make_fixtures.py` first');
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
    equal('title', await page.title(), 'Drone Inspection - Turbine, Solar, Crowd & Wildfire');
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

    console.log('\nCrowd image (yellow)');
    await page.locator('#file-input').setInputFiles(join(FIXTURES, 'crowd_yellow.png'));
    const crowd = await cardFor(page, 'crowd_yellow.png');
    await crowd.locator('.badge').waitFor({ timeout: 30000 });
    equal('severity badge', await crowd.locator('.badge').textContent(),
      `${EXPECT.crowd.severity} · score ${EXPECT.crowd.score}`);
    equal('detections', (await crowd.locator('.detections li').allTextContents())
      .map((t) => t.trim()).sort().join(' | '), EXPECT.crowd.detections.slice().sort().join(' | '));
    // Yellow scores on red too, so this passing means the gate compared four classes and
    // picked the right one, rather than falling through to the first that matched.
    check('the four-class gate routed yellow to crowd, not turbine',
      (await crowd.locator('.card__meta').first().textContent()).includes('Crowd'));
    check('the card says it marks regions rather than individuals',
      (await crowd.locator('.card__note').textContent()).includes('not individual people'));

    console.log('\nWildfire image (magenta)');
    await page.locator('#file-input').setInputFiles(join(FIXTURES, 'wildfire_magenta.png'));
    const wildfire = await cardFor(page, 'wildfire_magenta.png');
    await wildfire.locator('.badge').waitFor({ timeout: 30000 });
    equal('severity badge', await wildfire.locator('.badge').textContent(),
      `${EXPECT.wildfire.severity} · score ${EXPECT.wildfire.score}`);
    equal('detections', (await wildfire.locator('.detections li').allTextContents())
      .map((t) => t.trim()).sort().join(' | '), EXPECT.wildfire.detections.slice().sort().join(' | '));
    // A person at a fire outweighs the smoke around them, which is the whole reason the
    // weights exist rather than counting boxes.
    check('a person outweighs smoke in the score',
      EXPECT.wildfire.score === '4.20');
    check('the card says one frame cannot speak for the ground',
      (await wildfire.locator('.card__note').textContent()).includes('treeline'));

    console.log('\nInvalid image (green) - the rejection path');
    await page.locator('#file-input').setInputFiles(join(FIXTURES, 'invalid_green.png'));
    const invalid = await cardFor(page, 'invalid_green.png');
    await invalid.locator('.card__message').waitFor({ timeout: 30000 });
    equal('rejection message', (await invalid.locator('.card__message').textContent()).trim(),
      'This does not look like wind turbine blade, solar panel, crowd or wildfire.');
    check('card marked rejected', (await invalid.getAttribute('class')).includes('card--rejected'));
    check('no detections drawn on a rejected image',
      await invalid.locator('.detections').count() === 0);

    console.log('\nNon-image file');
    await page.locator('#file-input').setInputFiles(join(FIXTURES, 'not_an_image.txt'));
    await page.waitForTimeout(300);
    equal('banner', (await page.locator('#status-banner').textContent()).trim(),
      'Skipped 1 file that is neither an image nor a video.');
    // Five cards by now: turbine, solar, crowd, wildfire, and the rejected green image.
    equal('no card created for it', await page.locator('.card').count(), 5);

    console.log('\nSummary and export');
    const tiles = await page.locator('.summary__tiles .tile').allTextContents();
    check('four moderate results counted', /4Moderate/.test(tiles.join('')), tiles.join(' / '));
    check('one rejection counted', /1Rejected/.test(tiles.join('')), tiles.join(' / '));
    // Four subjects in one batch, so the verdict falls back to neutral wording rather than
    // describing a fire as damage.
    equal('overall verdict', (await page.locator('.summary__overall').textContent()).trim(),
      'Moderate findings');
    for (const id of ['export-json', 'export-images', 'print-report', 'clear']) {
      check(`${id} enabled`, await page.locator(`#${id}`).isEnabled());
    }

    const download = await Promise.all([
      page.waitForEvent('download'),
      page.locator('#export-json').click(),
    ]).then(([d]) => d);
    const exported = JSON.parse(await readFile(await download.path(), 'utf8'));
    equal('JSON export result count', exported.results.length, 5);
    equal('JSON export overall', exported.summary.overall, 'Moderate findings');
    equal('JSON export bbox is in original pixels',
      JSON.stringify(exported.results[0].detections[0].bbox), '[380,430,580,530]');

    // The training-data export, checked by unzipping it rather than by trusting the button.
    // It is written by hand (web/js/zip.js) because everything going in is already
    // compressed, so a real archiver would be a megabyte to save nothing - which means the
    // archive being readable at all is worth asserting.
    console.log('\nTraining-data export');
    const trainingDownload = await Promise.all([
      page.waitForEvent('download'),
      page.locator('#export-training').click(),
    ]).then(([d]) => d);
    const zipPath = await trainingDownload.path();
    const { execFileSync } = await import('node:child_process');
    const listing = execFileSync('python3', ['-c',
      'import sys, zipfile; print("\\n".join(zipfile.ZipFile(sys.argv[1]).namelist()))',
      zipPath]).toString().trim().split('\n');

    check('the archive opens', listing.length > 0, listing.join(' '));
    check('it holds an image for every analysed upload',
      listing.filter((n) => n.includes('/images/')).length === 4, listing.join(' '));
    check('and a label sidecar for each',
      listing.filter((n) => n.includes('/labels/')).length === 4, listing.join(' '));
    check('and the summary report_generator.py consumes',
      listing.some((n) => n.endsWith('inspection_summary.json')), listing.join(' '));

    const sidecar = JSON.parse(execFileSync('python3', ['-c',
      'import sys, zipfile\n'
      + 'z = zipfile.ZipFile(sys.argv[1])\n'
      + 'name = [n for n in z.namelist() if "/labels/" in n][0]\n'
      + 'sys.stdout.write(z.read(name).decode())',
      zipPath]).toString());

    // vlm_to_yolo.py needs the dimensions the boxes are relative to, and skips anything
    // nobody has checked. Both are the difference between a usable training set and a
    // detector taught to repeat a vision model's guesses.
    check('the sidecar carries the image dimensions', sidecar.width > 0 && sidecar.height > 0);
    check('it is marked unreviewed', sidecar.reviewed === false);
    check('it names the subject', typeof sidecar.domain === 'string' && sidecar.domain.length > 0);
    check('its boxes are in pixels of that image',
      sidecar.detections.every((d) => d.box[2] <= sidecar.width && d.box[3] <= sidecar.height),
      JSON.stringify(sidecar.detections));

    await page.locator('.card').first().scrollIntoViewIfNeeded();
    await page.screenshot({ path: join(ROOT, 'docs', 'web-app.png') });

    console.log('\nClear');
    await page.locator('#clear').click();
    equal('cards removed', await page.locator('.card').count(), 0);
    check('export disabled again', await page.locator('#export-json').isDisabled());

    // The engine picker changes where the images go, so the claim in the header has to
    // change with it. A page that still says "runs in your browser" while uploading to a
    // provider is not a cosmetic bug.
    console.log('\nEngine picker');
    check('starts on the on-device engine',
      await page.locator('#engine-key-field').isHidden());
    check('claims local processing by default',
      (await page.locator('#privacy-pill').textContent()).includes('Runs in your browser'));

    await page.locator('#engine-provider').selectOption('anthropic');
    check('asks for a key', await page.locator('#engine-key-field').isVisible());
    check('asks for a model', await page.locator('#engine-model-field').isVisible());
    equal('names the right key', (await page.locator('#engine-key-label').textContent()).trim(),
      'Anthropic API key');
    check('stops claiming local processing',
      (await page.locator('#privacy-pill').textContent()).includes('Anthropic'));
    check('says the images leave the device',
      (await page.locator('#engine-warning').textContent()).includes('sent to the provider'));
    check('refuses to refresh models with no key',
      await page.locator('#engine-refresh').isDisabled());

    await page.locator('#engine-provider').selectOption('gemini');
    equal('switching provider swaps the key label',
      (await page.locator('#engine-key-label').textContent()).trim(), 'Google AI Studio API key');
    check('switching provider clears the previous key',
      (await page.locator('#engine-key').inputValue()) === '');

    // Uploading with no key must say so, not fail silently or start a doomed request.
    await page.locator('#file-input').setInputFiles(join(FIXTURES, 'turbine_red.png'));
    check('refuses to analyse without a key, naming the one it wants',
      (await page.locator('#status-banner').textContent()).includes('Google AI Studio API key'));
    equal('and creates no card', await page.locator('.card').count(), 0);

    await page.locator('#engine-provider').selectOption('local');
    check('returning to local restores the privacy claim',
      (await page.locator('#privacy-pill').textContent()).includes('Runs in your browser'));

    console.log('\nInstallable app');
    const manifestResponse = await page.request.get(`http://127.0.0.1:${port}/manifest.webmanifest`);
    check('manifest is served', manifestResponse.ok());
    const appManifest = await manifestResponse.json();
    equal('opens without browser chrome', appManifest.display, 'standalone');
    for (const icon of appManifest.icons) {
      const iconResponse = await page.request.get(`http://127.0.0.1:${port}/${icon.src}`);
      check(`icon present: ${icon.src}`, iconResponse.ok());
    }
    const promptResponse = await page.request.get(`http://127.0.0.1:${port}/prompts/inspection.json`);
    check('the shared prompt is served', promptResponse.ok());

    console.log('\nRuntime');
    check('no uncaught page errors', consoleErrors.length === 0, consoleErrors.join('\n          '));

    // Degraded mode: the site must stay honest when weights have not been deployed yet,
    // which is exactly the state a fresh clone is in.
    console.log('\nDegraded mode - no models deployed');
    for (const model of ['gate.onnx', 'turbine.onnx', 'solar.onnx', 'crowd.onnx', 'wildfire.onnx']) {
      await rm(join(SITE, 'models', model), { force: true });
    }
    await page.goto(`http://127.0.0.1:${port}/?nomodels`, { waitUntil: 'networkidle' });
    const banner = (await page.locator('#status-banner').textContent()).trim();
    check('explains that no on-device model is deployed',
      banner.includes('No on-device model is deployed yet'), banner);
    check('points at the engine picker rather than at a build script',
      banner.includes('Analysis engine') && !banner.includes('export_onnx.py'), banner);
    check('page still renders rather than erroring', await page.locator('.empty').isVisible());

    // The state a fresh deploy is actually in, and the one that used to fail silently: no
    // local model, so the page pre-selects a provider and asks for a key instead of
    // accepting files and doing nothing with them.
    check('an API engine is pre-selected so the page is usable at all',
      (await page.locator('#engine-provider').inputValue()) !== 'local');
    check('the key field is showing', await page.locator('#engine-key-field').isVisible());

    // The drop zone must say what is missing on the very first paint, not only after
    // someone has dropped files into it and got nothing. A banner above the fold is easy
    // to scroll past, and the drop zone looked like a working uploader - which is how
    // "it does not upload" gets reported about an app behaving exactly as written.
    check('the drop zone says what is missing before anything is dropped on it',
      await page.locator('#drop-blocked').isVisible());
    check('and names the key it wants',
      (await page.locator('#drop-blocked').textContent()).includes('API key'));
    check('with a link to where that key comes from',
      (await page.locator('#engine-key-link').getAttribute('href') ?? '').startsWith('https://'));

    // Typing a key clears it, without a reload.
    await page.locator('#engine-key').fill('sk-ant-placeholder');
    check('the blocker clears the moment a key is entered',
      await page.locator('#drop-blocked').isHidden());
    await page.locator('#engine-key').fill('');
    check('and comes back when it is removed',
      await page.locator('#drop-blocked').isVisible());

    // Now force it back to the on-device engine, which has nothing to run, and upload.
    // The bug this replaces: the picker opened, files were chosen, and nothing happened.
    await page.locator('#engine-provider').selectOption('local');
    await page.locator('#file-input').setInputFiles(join(FIXTURES, 'turbine_red.png'));
    const refusal = (await page.locator('#status-banner').textContent()).trim();
    check('uploading with no engine to run it says so rather than doing nothing silently',
      refusal.includes('no model to run yet'), refusal);
    check('and says what to do about it',
      refusal.includes('API key'), refusal);
    equal('no card is created for a file it cannot analyse',
      await page.locator('.card').count(), 0);

    // Gate absent but a detector present: fall back to a manual choice instead of guessing.
    console.log('\nDegraded mode - detector without gate');
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

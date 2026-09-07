/**
 * Tests for the vision-model inspection path.
 *
 *   node tests/test_vlm.mjs
 *
 * No browser and no network. The three provider adapters are exercised against a stubbed
 * fetch, which checks the thing that actually breaks in production: whether each provider
 * gets the request shape it expects, and whether an error from it turns into a message a
 * user can act on. The canvas is stubbed too, so the module can be imported under node.
 *
 * These are the checks that matter here because the alternative - discovering a malformed
 * request by spending money on a real call and getting a 400 - is slow and costs credit.
 *
 * The prompt is fetched from web/prompts/inspection.json, the same file tools/vlm_inspect.py
 * reads, so the stubbed fetch has to serve it. That the module refuses to run without it is
 * itself checked below.
 */

let failures = 0;
let passes = 0;

function check(name, condition, detail = '') {
  if (condition) {
    passes += 1;
  } else {
    failures += 1;
    console.error(`  FAIL  ${name}${detail ? ` -- ${detail}` : ''}`);
    return;
  }
  console.log(`  ok    ${name}`);
}

async function throws(name, fn, pattern) {
  try {
    await fn();
    check(name, false, 'expected a throw, got none');
  } catch (error) {
    check(name, pattern.test(error.message), error.message);
  }
}

// --- stubs ------------------------------------------------------------------------------
// Only what vlm.js touches. A real canvas is not needed: the encoder's job here is to hand
// the adapter a base64 string, and whether the pixels are right is the browser test's
// problem, not this one's.
const DATA_URL = 'data:image/jpeg;base64,QUJD';

globalThis.document = {
  createElement() {
    return {
      width: 0,
      height: 0,
      getContext: () => ({ drawImage() {} }),
      toDataURL: () => DATA_URL,
    };
  },
};

const image = { width: 1000, height: 500 };

const { readFileSync } = await import('node:fs');
const PROMPT_DOC = readFileSync(
  new URL('../web/prompts/inspection.json', import.meta.url), 'utf8',
);

let lastRequest = null;
let servePrompt = true;

function stubFetch(status, body) {
  globalThis.fetch = async (url, init) => {
    if (String(url).includes('inspection.json')) {
      return servePrompt
        ? { ok: true, status: 200, text: async () => PROMPT_DOC, json: async () => JSON.parse(PROMPT_DOC) }
        : { ok: false, status: 404, text: async () => 'not found', json: async () => ({}) };
    }
    lastRequest = { url: String(url), init };
    return {
      ok: status >= 200 && status < 300,
      status,
      text: async () => (typeof body === 'string' ? body : JSON.stringify(body)),
    };
  };
}

stubFetch(200, {});

const REPLY = {
  asset: 'turbine',
  asset_reason: 'Three-blade horizontal-axis turbine against sky.',
  overall: 'One blade is severed at mid-span.',
  findings: [
    { label: 'Blade Severed', certainty: 'high', box: [0.1, 0.2, 0.4, 0.6], note: 'Ground the turbine.' },
    { label: 'leading edge erosion', certainty: 'low', box: [0.5, 0.5, 0.55, 0.52], note: '' },
    { label: 'corrosion', certainty: 'medium', box: null, note: 'Somewhere on the tower.' },
  ],
};

const { inspect, listModels, parseReply, toPixels, tidyLabel, PROVIDERS } =
  await import('../web/js/vlm.js');

// --- reply parsing ----------------------------------------------------------------------
console.log('\nReply parsing');
check('bare JSON', parseReply('{"a":1}').a === 1);
check('fenced JSON', parseReply('```json\n{"a":2}\n```').a === 2);
check('unlabelled fence', parseReply('```\n{"a":3}\n```').a === 3);
check('prose before and after', parseReply('Here you go:\n{"a":4}\nHope that helps.').a === 4);
check('nested braces survive the scan', parseReply('x {"a":{"b":5}} y').a.b === 5);
await throws('empty reply', async () => parseReply('   '), /empty response/);
await throws('no JSON at all', async () => parseReply('I cannot help with that.'), /did not return usable JSON/);

// --- box normalisation ------------------------------------------------------------------
// Compared with a tolerance rather than exactly: normalising to a fraction and back is two
// float divisions, so 1001 comes out as 1001.0000000000001. A tenth of a pixel is far below
// anything that matters for a drawn box.
const boxIs = (got, want) =>
  Array.isArray(got) && got.length === 4 && got.every((v, i) => Math.abs(v - want[i]) < 0.1);

console.log('\nBox normalisation');
check('fractions scale to pixels',
  boxIs(toPixels([0.1, 0.2, 0.4, 0.6], 1000, 500), [100, 100, 400, 300]));
check('0-1000 convention is rescaled',
  boxIs(toPixels([100, 200, 400, 600], 1000, 500), [100, 100, 400, 300]));
check('raw pixels are rescaled',
  boxIs(toPixels([1001, 1002, 2000, 2400], 4000, 4000), [1001, 1002, 2000, 2400]));
check('inverted corners are ordered',
  boxIs(toPixels([0.4, 0.6, 0.1, 0.2], 1000, 500), [100, 100, 400, 300]));
check('out of range is clamped to the frame',
  boxIs(toPixels([-0.5, -0.5, 1.5, 1.5], 1000, 500), [0, 0, 1000, 500]));
check('degenerate box is dropped', toPixels([0.5, 0.5, 0.5001, 0.5001], 1000, 500) === null);
check('wrong length is dropped', toPixels([0.1, 0.2, 0.3], 1000, 500) === null);
check('non-numeric is dropped', toPixels([0.1, 0.2, 'x', 0.6], 1000, 500) === null);
check('null is dropped', toPixels(null, 1000, 500) === null);

// --- labels -----------------------------------------------------------------------------
console.log('\nLabels');
check('spaces and case normalise', tidyLabel('Leading Edge Erosion') === 'leading_edge_erosion');
check('punctuation collapses', tidyLabel('crack -- severe!!') === 'crack_severe');
check('empty falls back', tidyLabel('') === 'defect');
check('undefined falls back', tidyLabel(undefined) === 'defect');

// --- provider request shapes --------------------------------------------------------------
console.log('\nAnthropic');
stubFetch(200, { content: [{ type: 'text', text: JSON.stringify(REPLY) }] });
let outcome = await inspect({ provider: 'anthropic', model: 'claude-opus-5', apiKey: 'k', image });
check('posts to the messages endpoint', lastRequest.url === 'https://api.anthropic.com/v1/messages');
check('sends the key in x-api-key', lastRequest.init.headers['x-api-key'] === 'k');
check('sends the api version', lastRequest.init.headers['anthropic-version'] === '2023-06-01');
check('opts in to browser access',
  lastRequest.init.headers['anthropic-dangerous-direct-browser-access'] === 'true');
check('sends a base64 image block',
  JSON.parse(lastRequest.init.body).messages[0].content[0].source.data === 'QUJD');

console.log('\nGemini');
stubFetch(200, { candidates: [{ content: { parts: [{ text: JSON.stringify(REPLY) }] } }] });
await inspect({ provider: 'gemini', model: 'gemini-2.5-pro', apiKey: 'k', image });
check('model id is in the path', lastRequest.url.includes('/models/gemini-2.5-pro:generateContent'));
check('key travels in a header, not the query string',
  lastRequest.init.headers['x-goog-api-key'] === 'k' && !lastRequest.url.includes('k'));
check('asks for a JSON response',
  JSON.parse(lastRequest.init.body).generationConfig.responseMimeType === 'application/json');
check('sends inline image data',
  JSON.parse(lastRequest.init.body).contents[0].parts[0].inline_data.data === 'QUJD');

console.log('\nOpenAI');
stubFetch(200, { choices: [{ message: { content: JSON.stringify(REPLY) } }] });
await inspect({ provider: 'openai', model: 'gpt-5', apiKey: 'k', image });
check('posts to chat completions', lastRequest.url === 'https://api.openai.com/v1/chat/completions');
check('sends a bearer token', lastRequest.init.headers.authorization === 'Bearer k');
check('requests a JSON object',
  JSON.parse(lastRequest.init.body).response_format.type === 'json_object');
check('sends the image as a data url',
  JSON.parse(lastRequest.init.body).messages[0].content[1].image_url.url === DATA_URL);

// --- outcome mapping ----------------------------------------------------------------------
console.log('\nOutcome');
check('asset is carried through', outcome.asset === 'turbine');
check('overall prose is carried through', outcome.overall === 'One blade is severed at mid-span.');
check('boxed defects become detections', outcome.detections.length === 2);
check('an unboxed defect is kept, not dropped', outcome.unlocated.length === 1);
check('the unboxed one is the right one', outcome.unlocated[0].label === 'corrosion');
check('labels are normalised', outcome.detections[0].label === 'blade_severed');
check('high certainty maps above medium',
  outcome.detections[0].confidence > outcome.detections[1].confidence);
check('the certainty band is preserved verbatim', outcome.detections[0].certainty === 'high');
check('a colour index is assigned', Number.isInteger(outcome.detections[0].classId));
check('the same label gets the same colour every time',
  outcome.detections[0].classId
  === (await inspect({ provider: 'openai', model: 'gpt-5', apiKey: 'k', image })).detections[0].classId);

stubFetch(200, { choices: [{ message: { content: '{"asset":"cat","findings":[]}' } }] });
outcome = await inspect({ provider: 'openai', model: 'gpt-5', apiKey: 'k', image });
check('an unknown asset value falls back to neither', outcome.asset === 'neither');

stubFetch(200, { choices: [{ message: { content: JSON.stringify({
  asset: 'crowd', asset_reason: 'Aerial view of a packed standing area.',
  overall: 'Dense packing against the stage-front barrier.',
  findings: [{ label: 'pressure against barrier', certainty: 'high',
    box: [0.2, 0.6, 0.8, 0.9], note: 'Relieve pressure at the front.' }],
}) } }] });
outcome = await inspect({ provider: 'openai', model: 'gpt-5', apiKey: 'k', image, domain: 'crowd' });
check('crowd is accepted as an asset', outcome.asset === 'crowd');
check('crowd findings become detections', outcome.detections.length === 1);
check('crowd labels are normalised',
  outcome.detections[0].label === 'pressure_against_barrier');

// --- errors -------------------------------------------------------------------------------
console.log('\nErrors');
stubFetch(401, { error: { message: 'invalid x-api-key' } });
await throws('401 names the key, not the request',
  () => inspect({ provider: 'anthropic', model: 'm', apiKey: 'bad', image }), /rejected the API key/);

stubFetch(404, { error: { message: 'model not found' } });
await throws('404 names the model and points at Refresh',
  () => inspect({ provider: 'openai', model: 'nope', apiKey: 'k', image }), /does not recognise that model/);

stubFetch(429, { error: { message: 'quota' } });
await throws('429 names rate limit or credit',
  () => inspect({ provider: 'gemini', model: 'm', apiKey: 'k', image }), /rate-limited or out of credit/);

stubFetch(503, 'upstream unavailable');
await throws('5xx says it is the provider, not you',
  () => inspect({ provider: 'gemini', model: 'm', apiKey: 'k', image }), /server error/);

stubFetch(200, { content: [{ type: 'text', text: 'sorry, I cannot see images' }] });
await throws('a refusal is an error, not an empty clean result',
  () => inspect({ provider: 'anthropic', model: 'm', apiKey: 'k', image }), /did not return usable JSON/);

await throws('a missing key is caught before the request',
  () => inspect({ provider: 'anthropic', model: 'm', apiKey: '', image }), /Enter an API key/);
await throws('a missing model is caught before the request',
  () => inspect({ provider: 'anthropic', model: '', apiKey: 'k', image }), /Choose a model/);
await throws('an unknown provider is caught',
  () => inspect({ provider: 'llama', model: 'm', apiKey: 'k', image }), /Unknown provider/);

// --- model listing --------------------------------------------------------------------------
console.log('\nModel listing');
stubFetch(200, { data: [{ id: 'claude-opus-5', display_name: 'Claude Opus 5' }] });
check('anthropic list is mapped', (await listModels('anthropic', 'k'))[0].label === 'Claude Opus 5');

stubFetch(200, {
  models: [
    { name: 'models/gemini-2.5-pro', displayName: 'Gemini 2.5 Pro', supportedGenerationMethods: ['generateContent'] },
    { name: 'models/text-embedding-004', displayName: 'Embedding', supportedGenerationMethods: ['embedContent'] },
  ],
});
let listed = await listModels('gemini', 'k');
check('gemini strips the models/ prefix', listed[0].id === 'gemini-2.5-pro');
check('gemini drops models that cannot generate', listed.length === 1);

stubFetch(200, {
  data: [
    { id: 'gpt-5' }, { id: 'gpt-4o' }, { id: 'whisper-1' },
    { id: 'text-embedding-3-small' }, { id: 'dall-e-3' }, { id: 'gpt-4o-audio-preview' },
  ],
});
listed = await listModels('openai', 'k');
check('openai keeps only chat-capable ids',
  listed.map((m) => m.id).join(',') === 'gpt-4o,gpt-5', listed.map((m) => m.id).join(','));

stubFetch(200, { data: [] });
await throws('an empty list is an error, not a silent empty dropdown',
  () => listModels('anthropic', 'k'), /returned no models/);

// --- the shared prompt ------------------------------------------------------------------
console.log('\nShared prompt');
const doc = JSON.parse(PROMPT_DOC);
check('has a brief for every domain',
  ['turbine', 'solar', 'crowd', 'auto'].every((k) => (doc.domains[k] ?? '').length > 100));
check('the schema names every key the parser reads',
  ['asset', 'asset_reason', 'findings', 'label', 'certainty', 'box', 'note', 'overall']
    .every((k) => doc.schema.includes(k)));
check('the schema states the box convention', /fractions of image width and height/.test(doc.schema));
check('an empty result is described as a null result, not as reassurance',
  /It is not a statement that the asset is sound or that the scene is without risk/.test(doc.schema));
check('the crowd brief refuses to put a number on people',
  /never as a number of\npeople/.test(doc.domains.crowd));
check('the crowd brief boxes regions rather than individuals',
  /Box the region, never the individual/.test(doc.domains.crowd));
check('crowd is a recognised asset', /"crowd"/.test(doc.schema));
check('the solar brief refuses thermal-only faults', /thermal infrared/.test(doc.domains.solar));
check('the turbine brief names structural failure', /severed/.test(doc.domains.turbine));
check('certainty bands are ordered', doc.certainty.high > doc.certainty.medium
  && doc.certainty.medium > doc.certainty.low);

stubFetch(200, { choices: [{ message: { content: JSON.stringify(REPLY) } }] });
await inspect({ provider: 'openai', model: 'gpt-5', apiKey: 'k', image, domain: 'turbine' });
const sentPrompt = JSON.parse(lastRequest.init.body).messages[0].content[0].text;
check('the turbine brief reaches the provider', sentPrompt.includes(doc.domains.turbine));
check('the schema is appended to it', sentPrompt.includes(doc.schema));

await inspect({ provider: 'openai', model: 'gpt-5', apiKey: 'k', image, domain: 'nonsense' });
check('an unknown domain falls back to auto',
  JSON.parse(lastRequest.init.body).messages[0].content[0].text.includes(doc.domains.auto));

// --- catalogue ---------------------------------------------------------------------------------
console.log('\nCatalogue');
for (const [key, provider] of Object.entries(PROVIDERS)) {
  check(`${key} has a fallback model list`, provider.models.length > 0);
  check(`${key} explains where its key comes from`, Boolean(provider.keyHint));
}

console.log(`\n${passes} passed, ${failures} failed`);
process.exit(failures ? 1 : 0);

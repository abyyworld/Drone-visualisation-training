/**
 * Service worker: makes the app installable and lets it open without a connection.
 *
 * WHY
 *     The tablet this runs on is a drone controller. It is taken to a wind farm, where
 *     there is often no usable mobile signal, and it is expected to behave like an app
 *     rather than a bookmark. A service worker plus web/manifest.webmanifest is what turns
 *     the page into something with an icon on the home screen, its own window and no
 *     browser chrome - with no APK to build, sign, or persuade an Android launcher to
 *     accept. See docs/INSTALL-tablet.md.
 *
 * WHAT WORKS OFFLINE
 *     The app shell and the on-device models, once they have been downloaded on a
 *     connection. Analysis with the on-device engine then runs with the tablet in
 *     flight mode. The API engines cannot: they are a network round trip by definition,
 *     and the app says so rather than queueing work that will never send.
 *
 * CACHING STRATEGY
 *     App shell: cache first, revalidated in the background. It is small, it changes only
 *     on deploy, and a fast cold start on a tablet matters more than being one deploy
 *     behind for a few seconds.
 *
 *     Models and the runtime WASM: cache first, no revalidation. They are megabytes and
 *     immutable for a given filename - re-fetching them on a metered connection at a site
 *     would be an expensive way to get identical bytes.
 *
 *     Everything else, including every provider API call: straight to the network,
 *     never cached. An inspection result must never be served from a cache.
 *
 * BUMPING THE VERSION
 *     Change CACHE_VERSION on any deploy that changes the app shell. Old caches are
 *     deleted on activate, so a stale shell cannot outlive a release.
 */

const CACHE_VERSION = 'v3';
const SHELL_CACHE = `inspection-shell-${CACHE_VERSION}`;
const ASSET_CACHE = `inspection-assets-${CACHE_VERSION}`;

const SHELL = [
  './',
  'index.html',
  'css/app.css',
  'js/app.js',
  'js/runtime.js',
  'js/preprocess.js',
  'js/gate.js',
  'js/detect.js',
  'js/severity.js',
  'js/render.js',
  'js/vlm.js',
  'js/video.js',
  'js/formats.js',
  'prompts/inspection.json',
  'models/manifest.json',
  'manifest.webmanifest',
  'icons/icon-192.png',
  'icons/icon-512.png',
];

// Big and immutable for a given name. Cached on first use rather than at install, because
// pre-fetching 30 MB of weights the moment someone opens the page would be rude.
const IMMUTABLE = /\.(onnx|wasm|mjs)$|\/ort\//;

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(SHELL_CACHE);
    // Individually, not addAll: addAll rejects the whole install if any single file 404s,
    // and a deploy missing one optional file should not leave the app uninstallable.
    await Promise.all(SHELL.map((url) => cache.add(url).catch(() => {})));
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keep = new Set([SHELL_CACHE, ASSET_CACHE]);
    await Promise.all(
      (await caches.keys()).filter((key) => !keep.has(key)).map((key) => caches.delete(key)),
    );
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;

  const url = new URL(request.url);
  // Anything off this origin is a provider API call or a CDN script. Never cached: a
  // cached inspection response would be a wrong answer served fast.
  if (url.origin !== self.location.origin) return;

  if (IMMUTABLE.test(url.pathname)) {
    event.respondWith(cacheFirst(request, ASSET_CACHE));
    return;
  }
  event.respondWith(staleWhileRevalidate(request, SHELL_CACHE));
});

async function cacheFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  const hit = await cache.match(request);
  if (hit) return hit;

  const response = await fetch(request);
  if (response.ok) cache.put(request, response.clone());
  return response;
}

async function staleWhileRevalidate(request, cacheName) {
  const cache = await caches.open(cacheName);
  const hit = await cache.match(request);

  const fresh = fetch(request)
    .then((response) => {
      if (response.ok) cache.put(request, response.clone());
      return response;
    })
    .catch(() => null);

  if (hit) return hit;

  const response = await fresh;
  if (response) return response;

  // Offline with nothing cached. A navigation gets the shell if it is there; anything
  // else gets an honest failure rather than a blank page.
  if (request.mode === 'navigate') {
    const shell = await cache.match('index.html');
    if (shell) return shell;
  }
  return new Response('Offline, and this has not been cached yet.', {
    status: 503,
    headers: { 'content-type': 'text/plain' },
  });
}

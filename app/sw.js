/**
 * Service worker: the offline app shell.
 *
 * The station is a laptop on a fire ground with no internet and, often, no
 * route to anything but itself. This worker exists so that a tablet that has
 * opened the app once can open it again while the station is still booting, or
 * while the WiFi is being moved, and be sitting at the connection screen
 * instead of at a browser error page.
 *
 * Two rules matter more than any caching cleverness:
 *
 *  1. **Signalling and configuration are never cached.** A cached SDP answer
 *     would hand a second tablet the first tablet's peer id; cached overlay
 *     parameters would mean a tablet enforcing last week's staleness limit.
 *     Those requests are passed straight through, and they are not even
 *     inspected on the way.
 *  2. **The worker is optional.** Registration is allowed to fail (a
 *     self-signed LAN certificate is exactly the condition under which it
 *     might), and nothing in the app depends on it having succeeded.
 *
 * There is no offline fallback page that says anything about the scene, and
 * there is nothing cached that could be replayed as a detection. The cache
 * holds the shell only: markup, style, modules, icons.
 */

/** Bump on every shell change; the old cache is deleted on activate. */
const CACHE = 'wildfire-watch-shell-v1';

/** Paths that must never be served from, or written to, the cache. */
const NEVER_CACHE = ['/webrtc/', '/config', '/peers', '/offer', '/close'];

const SHELL = [
  './',
  'index.html',
  'manifest.webmanifest',
  'css/style.css',
  'js/app.js',
  'js/wire.js',
  'js/sync.js',
  'js/overlay.js',
  'js/connection.js',
  'icons/icon-192.png',
  'icons/icon-512.png',
  'icons/icon-maskable-512.png',
  'icons/apple-touch-icon-180.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE);
      // Individually, not addAll: one missing optional file (an icon on a
      // trimmed deployment) must not fail the whole installation and leave the
      // tablet with no shell at all.
      await Promise.all(
        SHELL.map(async (path) => {
          try {
            await cache.add(new Request(path, { cache: 'reload' }));
          } catch (err) {
            console.warn('shell asset not cached:', path, err);
          }
        }),
      );
      // The new shell is wanted immediately: the alternative is a tablet
      // running last week's overlay logic until every tab is closed, which on
      // a home-screen app can be days.
      await self.skipWaiting();
    })(),
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      const names = await caches.keys();
      await Promise.all(names.filter((name) => name !== CACHE).map((name) => caches.delete(name)));
      await self.clients.claim();
    })(),
  );
});

/**
 * True for requests that must always go to the network.
 *
 * @param {URL} url The request URL.
 * @returns {boolean} Whether to bypass the cache entirely.
 */
function isLive(url) {
  return NEVER_CACHE.some((fragment) => url.pathname.includes(fragment));
}

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  if (isLive(url)) return; // straight to the network, uninspected

  if (request.mode === 'navigate') {
    // Network first for the page itself, so a tablet that can reach the
    // station always gets the build the station is serving.
    event.respondWith(
      (async () => {
        try {
          const fresh = await fetch(request);
          const cache = await caches.open(CACHE);
          cache.put('index.html', fresh.clone());
          return fresh;
        } catch (err) {
          const cached = (await caches.match('index.html')) || (await caches.match('./'));
          if (cached) return cached;
          throw err;
        }
      })(),
    );
    return;
  }

  // Static assets: cache first for an instant start, with a background
  // refresh so the next start picks up a redeployed station.
  event.respondWith(
    (async () => {
      const cached = await caches.match(request);
      const network = fetch(request)
        .then(async (response) => {
          if (response && response.ok) {
            const cache = await caches.open(CACHE);
            cache.put(request, response.clone());
          }
          return response;
        })
        .catch((err) => {
          if (cached) return cached;
          throw err;
        });
      return cached || network;
    })(),
  );
});

self.addEventListener('message', (event) => {
  if (event.data === 'skip-waiting') void self.skipWaiting();
});

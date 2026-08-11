/**
 * Service worker for the offline shell.
 *
 * Deliberately narrow. Model weights are already cached by transformers.js and
 * WebLLM in their own Cache Storage buckets, and re-caching hundreds of
 * megabytes here would double the storage cost for no gain. What this handles is
 * the app shell and the bundled corpora, so Worlds 1 to 5 open with no network
 * once their models have been fetched at least once (Section 9 / Phase 9).
 */

const VERSION = 'v1';
const SHELL_CACHE = `ail-shell-${VERSION}`;
const ASSET_CACHE = `ail-assets-${VERSION}`;

/** Fetched up front so the first offline launch has something to render. */
const SHELL_URLS = ['/', '/map', '/onboarding', '/manifest.webmanifest', '/icon.svg'];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(SHELL_CACHE)
      // One missing URL must not abandon the whole precache.
      .then((cache) => Promise.allSettled(SHELL_URLS.map((url) => cache.add(url))))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((key) => key.startsWith('ail-') && !key.endsWith(VERSION))
            .map((key) => caches.delete(key))
        )
      )
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET') return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  // The API is never cached: stale activity or a stale cloud response would be
  // worse than an honest failure, and the client already queues offline writes.
  if (url.pathname.startsWith('/api/')) return;

  // Corpora and static assets change only on deploy — cache first.
  if (url.pathname.startsWith('/corpora/') || url.pathname.startsWith('/_next/static/')) {
    event.respondWith(cacheFirst(request, ASSET_CACHE));
    return;
  }

  // Navigations: try the network so a deploy is picked up, fall back to the
  // cached shell so the app still opens with no connection.
  if (request.mode === 'navigate') {
    event.respondWith(networkFirst(request, SHELL_CACHE));
    return;
  }

  event.respondWith(cacheFirst(request, ASSET_CACHE));
});

async function cacheFirst(request, cacheName) {
  const cached = await caches.match(request);
  if (cached) return cached;

  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(cacheName);
      void cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    const fallback = await caches.match(request);
    if (fallback) return fallback;
    throw error;
  }
}

async function networkFirst(request, cacheName) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(cacheName);
      void cache.put(request, response.clone());
    }
    return response;
  } catch {
    const cached = (await caches.match(request)) ?? (await caches.match('/map'));
    if (cached) return cached;
    return new Response('Offline and this page has not been cached yet.', {
      status: 503,
      headers: { 'Content-Type': 'text/plain' },
    });
  }
}

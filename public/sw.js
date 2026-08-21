/**
 * Service worker for the offline shell.
 *
 * Deliberately narrow. Model weights are already cached by transformers.js and
 * WebLLM in their own Cache Storage buckets, and re-caching hundreds of
 * megabytes here would double the storage cost for no gain. What this handles is
 * the app shell and the bundled corpora, so Worlds 1 to 5 open with no network
 * once their models have been fetched at least once (Section 9 / Phase 9).
 */

const VERSION = 'v2';
const SHELL_CACHE = `ail-shell-${VERSION}`;
const ASSET_CACHE = `ail-assets-${VERSION}`;

/**
 * Every chapter route is precached explicitly, not left to `networkFirst`'s
 * cache-as-you-go behaviour below. A real player always reaches a chapter by
 * clicking through `/map` — a client-side transition — and Next's RSC fetch
 * for that transition never arrives here as a `navigate`-mode request, so it
 * never gets cached as a side effect. Verified for real (A4,
 * `plan-docs/REMAINING-WORK.md`): without this list, going offline and
 * reloading a chapter reached that way silently served the cached `/map`
 * shell instead, under a URL bar that still read the chapter's own path.
 * Hardcoded rather than read from the manifest at request time — this file
 * is a plain static asset with no build step of its own — so a 23rd chapter
 * needs a line added here too.
 */
const CHAPTER_URLS = [
  '/world/1/chapter/1-1-vectors',
  '/world/1/chapter/1-2-vector-arithmetic',
  '/world/1/chapter/1-3-similarity-distance',
  '/world/1/chapter/1-4-tokenization',
  '/world/1/chapter/1-5-probability',
  '/world/2/chapter/2-1-perceptron',
  '/world/2/chapter/2-2-loss-functions',
  '/world/2/chapter/2-3-gradient-descent',
  '/world/2/chapter/2-4-overfitting',
  '/world/3/chapter/3-1-neurons-activations',
  '/world/3/chapter/3-2-layers-forward-pass',
  '/world/3/chapter/3-3-backpropagation',
  '/world/3/chapter/3-4-training-dynamics',
  '/world/4/chapter/4-1-ngrams',
  '/world/4/chapter/4-2-recurrence-memory',
  '/world/4/chapter/4-3-sampling-strategies',
  '/world/5/chapter/5-1-positional-encoding',
  '/world/5/chapter/5-2-self-attention',
  '/world/5/chapter/5-3-multi-head-attention',
  '/world/5/chapter/5-4-residuals-layernorm',
  '/world/5/chapter/5-5-full-transformer',
  '/world/6/chapter/6-1-inspector-chat',
  '/world/7/chapter/7-1-retrieval',
  '/world/7/chapter/7-2-grounded-generation',
  '/world/7/chapter/7-3-tool-calling',
  '/world/7/chapter/7-4-agent-loop',
  '/world/8/chapter/8-1-quantization',
  '/world/8/chapter/8-2-context-length',
  '/world/8/chapter/8-3-calibration-hallucination',
  '/world/8/chapter/8-4-red-teaming',
];

/** Fetched up front so the first offline launch has something to render. */
const SHELL_URLS = ['/', '/map', '/onboarding', '/manifest.webmanifest', '/icon.svg', ...CHAPTER_URLS];

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
    // `ignoreSearch` matters for shared chapter links (`?via=share`) — without
    // it, the exact-match lookup misses the precached chapter entry (which
    // has no query string) and silently falls back to the `/map` shell, the
    // same bug class this file already fixed once for the plain-URL case.
    const cached =
      (await caches.match(request, { ignoreSearch: true })) ?? (await caches.match('/map'));
    if (cached) return cached;
    return new Response('Offline and this page has not been cached yet.', {
      status: 503,
      headers: { 'Content-Type': 'text/plain' },
    });
  }
}

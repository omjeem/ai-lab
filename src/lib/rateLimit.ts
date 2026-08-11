/**
 * IP-based sliding-window rate limiting (Section 11.4).
 *
 * The public routes carry no per-user auth, so without this anyone can burn the
 * Ollama Cloud budget or fill the activity collection. In-memory per instance,
 * which is the right trade for this: it costs nothing, survives a restart badly,
 * and is a hurdle rather than a wall.
 */

interface Window {
  hits: number[];
}

const windows = new Map<string, Window>();
const WINDOW_MS = 60_000;
/** Sweep threshold so a long-running process does not accumulate dead keys. */
const MAX_TRACKED_KEYS = 10_000;

export interface RateLimitResult {
  allowed: boolean;
  remaining: number;
  /** Seconds until the oldest hit falls out of the window. */
  retryAfterSeconds: number;
}

export function rateLimit(key: string, limitPerMinute: number): RateLimitResult {
  const now = Date.now();
  const cutoff = now - WINDOW_MS;

  if (windows.size > MAX_TRACKED_KEYS) sweep(cutoff);

  const window = windows.get(key) ?? { hits: [] };
  window.hits = window.hits.filter((t) => t > cutoff);

  if (window.hits.length >= limitPerMinute) {
    windows.set(key, window);
    const oldest = window.hits[0] ?? now;
    return {
      allowed: false,
      remaining: 0,
      retryAfterSeconds: Math.max(1, Math.ceil((oldest + WINDOW_MS - now) / 1000)),
    };
  }

  window.hits.push(now);
  windows.set(key, window);
  return {
    allowed: true,
    remaining: Math.max(0, limitPerMinute - window.hits.length),
    retryAfterSeconds: 0,
  };
}

function sweep(cutoff: number): void {
  for (const [key, window] of windows) {
    const live = window.hits.filter((t) => t > cutoff);
    if (live.length === 0) windows.delete(key);
    else window.hits = live;
  }
}

/**
 * Best-effort client IP.
 *
 * Reads the proxy headers a deployment normally sits behind, preferring the
 * leftmost entry in `x-forwarded-for` — the original client.
 */
export function clientIpFrom(headers: Headers): string | null {
  const forwarded = headers.get('x-forwarded-for');
  if (forwarded) {
    const first = forwarded.split(',')[0]?.trim();
    if (first) return first;
  }
  return (
    headers.get('cf-connecting-ip') ??
    headers.get('x-real-ip') ??
    headers.get('x-vercel-forwarded-for') ??
    null
  );
}

export function readLimit(name: string, fallback: number): number {
  const raw = process.env[name];
  const parsed = raw ? Number(raw) : NaN;
  return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : fallback;
}

/** Test seam. */
export function resetRateLimits(): void {
  windows.clear();
}

/**
 * Where a visitor's first request came from, derived from the standard
 * `Referer` header — no `?source=` query param bridging, no new cookie.
 */
export function deriveSource(referrer: string | null, host: string | null): string {
  if (!referrer) return 'Direct · direct';

  let referrerHost: string;
  try {
    referrerHost = new URL(referrer).hostname;
  } catch {
    return 'Direct · direct';
  }

  if (host && referrerHost === stripPort(host)) return 'Internal · internal';
  return `${referrerHost} · external`;
}

function stripPort(host: string): string {
  return host.split(':')[0] ?? host;
}

/**
 * Optional IP geolocation enrichment.
 *
 * Section 8 is explicit that no paid third-party API should be called without
 * the operator configuring their own: with `GEO_LOOKUP_URL` unset this does
 * nothing at all, and onboarding still stores the IP and headers.
 */
import type { RequestEnrichment } from '@/types/user';

const LOOKUP_TIMEOUT_MS = 2500;

interface GeoPayload {
  country_name?: string;
  country?: string;
  region?: string;
  region_name?: string;
  city?: string;
}

export function isGeoLookupConfigured(): boolean {
  return Boolean(process.env.GEO_LOOKUP_URL);
}

/** Private and loopback ranges have no useful geolocation. */
export function isPrivateAddress(ip: string): boolean {
  return (
    ip === '::1' ||
    ip.startsWith('127.') ||
    ip.startsWith('10.') ||
    ip.startsWith('192.168.') ||
    /^172\.(1[6-9]|2\d|3[01])\./.test(ip) ||
    ip.startsWith('fc') ||
    ip.startsWith('fd')
  );
}

export async function lookupGeo(ip: string | null): Promise<RequestEnrichment['geo']> {
  const template = process.env.GEO_LOOKUP_URL;
  if (!template || !ip || isPrivateAddress(ip)) return null;

  const url = template.replace('{ip}', encodeURIComponent(ip));
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), LOOKUP_TIMEOUT_MS);

  try {
    const apiKey = process.env.GEO_LOOKUP_API_KEY;
    const response = await fetch(url, {
      signal: controller.signal,
      headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : undefined,
    });
    if (!response.ok) return null;

    const payload = (await response.json()) as GeoPayload;
    return {
      country: payload.country_name ?? payload.country ?? null,
      region: payload.region_name ?? payload.region ?? null,
      city: payload.city ?? null,
      source: new URL(url).host,
    };
  } catch {
    // Enrichment is a nice-to-have; onboarding must never fail over it.
    return null;
  } finally {
    clearTimeout(timer);
  }
}

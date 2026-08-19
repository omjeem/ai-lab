import { describe, it, expect } from 'vitest';
import { geoFromVercelHeaders } from '@/lib/geoLookup';

describe('geoFromVercelHeaders', () => {
  it('returns null when Vercel geo headers are absent (e.g. off-Vercel or local dev)', () => {
    expect(geoFromVercelHeaders(new Headers())).toBeNull();
  });

  it('reads country/region/city/coords/timezone straight off the edge headers', () => {
    const headers = new Headers({
      'x-vercel-ip-country': 'US',
      'x-vercel-ip-country-region': 'CA',
      'x-vercel-ip-city': 'San%20Francisco',
      'x-vercel-ip-latitude': '37.7749',
      'x-vercel-ip-longitude': '-122.4194',
      'x-vercel-ip-timezone': 'America/Los_Angeles',
    });

    expect(geoFromVercelHeaders(headers)).toEqual({
      country: 'US',
      region: 'CA',
      city: 'San Francisco',
      latitude: '37.7749',
      longitude: '-122.4194',
      timezone: 'America/Los_Angeles',
      source: 'vercel',
    });
  });

  it('falls back gracefully when only the country header is present', () => {
    const headers = new Headers({ 'x-vercel-ip-country': 'DE' });
    expect(geoFromVercelHeaders(headers)).toEqual({
      country: 'DE',
      region: null,
      city: null,
      latitude: null,
      longitude: null,
      timezone: null,
      source: 'vercel',
    });
  });
});

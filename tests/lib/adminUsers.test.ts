import { describe, it, expect } from 'vitest';
import { shapeAdminUserRow } from '@/lib/adminUsers';
import type { UserDocument } from '@/types/user';

function fabricateDoc(overrides: Partial<UserDocument> = {}): UserDocument {
  return {
    userId: 'user-1',
    name: 'Ada',
    firstSeen: new Date('2026-01-01T00:00:00.000Z'),
    lastSeen: new Date('2026-01-02T00:00:00.000Z'),
    firstReferrer: null,
    firstSource: 'Direct · direct',
    client: undefined,
    enrichment: {
      ip: '1.2.3.4',
      userAgent: 'Mozilla/5.0',
      referrer: null,
      acceptLanguage: 'en-US',
      host: 'ailab.example.com',
      geo: null,
      device: null,
    },
    ...overrides,
  };
}

describe('shapeAdminUserRow', () => {
  it('maps a bare document with no client/geo/device data to all-null optional fields', () => {
    const row = shapeAdminUserRow(fabricateDoc(), 0, undefined);
    expect(row.userId).toBe('user-1');
    expect(row.eventCount).toBe(0);
    expect(row.country).toBeNull();
    expect(row.landingPage).toBeNull();
    expect(row.screen).toBeNull();
    expect(row.languages).toBeNull();
  });

  it('pulls geo, device, and host straight through from enrichment', () => {
    const row = shapeAdminUserRow(
      fabricateDoc({
        enrichment: {
          ip: '1.2.3.4',
          userAgent: 'Mozilla/5.0',
          referrer: null,
          acceptLanguage: 'en-US',
          host: 'ailab.example.com',
          geo: {
            country: 'IN',
            region: 'MH',
            city: 'Pune',
            source: 'vercel',
            latitude: '18.52',
            longitude: '73.85',
            timezone: 'Asia/Kolkata',
          },
          device: {
            browser: 'Chrome',
            browserVersion: '120',
            os: 'Windows',
            osVersion: '10',
            deviceType: 'desktop',
            isBot: false,
          },
        },
      }),
      3,
      undefined
    );

    expect(row.country).toBe('IN');
    expect(row.city).toBe('Pune');
    expect(row.latitude).toBe('18.52');
    expect(row.longitude).toBe('73.85');
    expect(row.geoTimezone).toBe('Asia/Kolkata');
    expect(row.host).toBe('ailab.example.com');
    expect(row.browser).toBe('Chrome');
    expect(row.browserVersion).toBe('120');
    expect(row.deviceType).toBe('desktop');
    expect(row.eventCount).toBe(3);
  });

  it('joins screen/viewport into "WxH" and languages into a comma list', () => {
    const row = shapeAdminUserRow(
      fabricateDoc({
        client: {
          screenWidth: 1366,
          screenHeight: 768,
          viewportWidth: 1366,
          viewportHeight: 641,
          devicePixelRatio: 1,
          languages: ['en-US', 'en'],
          platform: 'Win32',
          touchSupport: false,
          deviceMemoryGb: 16,
          cpuCores: 8,
          networkType: '4g',
          colorScheme: 'light',
          timezone: 'Asia/Kolkata',
        },
      }),
      0,
      undefined
    );

    expect(row.screen).toBe('1366×768');
    expect(row.viewport).toBe('1366×641');
    expect(row.languages).toBe('en-US, en');
    expect(row.platform).toBe('Win32');
    expect(row.touchSupport).toBe(false);
    expect(row.deviceMemoryGb).toBe(16);
    expect(row.cpuCores).toBe(8);
    expect(row.networkType).toBe('4g');
    expect(row.colorScheme).toBe('light');
  });

  it('uses the page_viewed edges for landing/last page when supplied', () => {
    const row = shapeAdminUserRow(fabricateDoc(), 5, {
      landingPage: '/onboarding',
      landingTitle: 'AI Learning Lab',
      lastPage: '/map',
    });

    expect(row.landingPage).toBe('/onboarding');
    expect(row.landingTitle).toBe('AI Learning Lab');
    expect(row.lastPage).toBe('/map');
  });

  it('carries firstSource through as `source`', () => {
    const row = shapeAdminUserRow(fabricateDoc({ firstSource: 'google.com · external' }), 0, undefined);
    expect(row.source).toBe('google.com · external');
  });
});

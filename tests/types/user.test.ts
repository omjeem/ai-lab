import { describe, it, expect } from 'vitest';
import { createUserRequestSchema } from '@/types/user';

describe('createUserRequestSchema client block', () => {
  const base = { userId: 'a-valid-user-id', name: 'Ada' };

  it('accepts a full client payload including the newer device/locale signals', () => {
    const result = createUserRequestSchema.safeParse({
      ...base,
      client: {
        timezoneOffsetMinutes: 330,
        timezone: 'Asia/Kolkata',
        screenWidth: 1366,
        screenHeight: 768,
        viewportWidth: 1366,
        viewportHeight: 641,
        devicePixelRatio: 1,
        language: 'en-US',
        languages: ['en-US', 'en'],
        platform: 'Win32',
        touchSupport: false,
        deviceMemoryGb: 16,
        cpuCores: 8,
        networkType: '4g',
        colorScheme: 'light',
      },
    });
    expect(result.success).toBe(true);
  });

  it('still accepts no client block at all (offline-first onboarding)', () => {
    expect(createUserRequestSchema.safeParse(base).success).toBe(true);
  });

  it('rejects a colorScheme outside the light/dark enum', () => {
    const result = createUserRequestSchema.safeParse({
      ...base,
      client: { colorScheme: 'sepia' },
    });
    expect(result.success).toBe(false);
  });
});

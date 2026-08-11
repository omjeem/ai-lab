import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  detectCapabilities,
  hasRoomFor,
  resetCapabilitiesCache,
  type DeviceCapabilities,
} from '@/lib/deviceCapabilities';

beforeEach(() => resetCapabilitiesCache());
afterEach(() => vi.unstubAllGlobals());

function stubNavigator(overrides: Record<string, unknown>) {
  vi.stubGlobal('navigator', overrides);
}

describe('detectCapabilities', () => {
  it('chooses WebGPU when an adapter is available', async () => {
    stubNavigator({ gpu: { requestAdapter: async () => ({ name: 'fake' }) } });
    const caps = await detectCapabilities(true);
    expect(caps.backend).toBe('webgpu');
    expect(caps.webgpu).toBe(true);
    expect(caps.reason).toBeNull();
  });

  it('falls back to WASM when the browser has no WebGPU at all', async () => {
    stubNavigator({});
    const caps = await detectCapabilities(true);
    expect(caps.backend).toBe('wasm');
    expect(caps.reason).toMatch(/does not expose WebGPU/);
  });

  it('falls back to WASM when no adapter is returned', async () => {
    stubNavigator({ gpu: { requestAdapter: async () => null } });
    const caps = await detectCapabilities(true);
    expect(caps.backend).toBe('wasm');
    expect(caps.reason).toMatch(/No WebGPU adapter/);
  });

  it('falls back to WASM when the adapter request throws', async () => {
    stubNavigator({
      gpu: {
        requestAdapter: async () => {
          throw new Error('driver exploded');
        },
      },
    });
    const caps = await detectCapabilities(true);
    expect(caps.backend).toBe('wasm');
    expect(caps.reason).toMatch(/driver exploded/);
  });

  it('reads the storage estimate when the browser offers one', async () => {
    stubNavigator({
      gpu: { requestAdapter: async () => ({}) },
      storage: { estimate: async () => ({ quota: 2_000_000_000, usage: 500_000_000 }) },
    });
    const caps = await detectCapabilities(true);
    expect(caps.storageQuotaMB).toBe(2000);
    expect(caps.storageUsedMB).toBe(500);
  });

  it('copes with a browser that refuses to estimate storage', async () => {
    stubNavigator({
      gpu: { requestAdapter: async () => ({}) },
      storage: {
        estimate: async () => {
          throw new Error('denied');
        },
      },
    });
    const caps = await detectCapabilities(true);
    expect(caps.storageQuotaMB).toBeNull();
  });

  it('caches the result until forced to re-detect', async () => {
    let calls = 0;
    stubNavigator({
      gpu: {
        requestAdapter: async () => {
          calls++;
          return {};
        },
      },
    });

    await detectCapabilities(true);
    await detectCapabilities();
    expect(calls).toBe(1);

    await detectCapabilities(true);
    expect(calls).toBe(2);
  });
});

describe('hasRoomFor', () => {
  const base: DeviceCapabilities = {
    backend: 'wasm',
    webgpu: false,
    reason: null,
    storageQuotaMB: 1000,
    storageUsedMB: 100,
  };

  it('allows a download that fits with headroom to spare', () => {
    expect(hasRoomFor(base, 100)).toBe(true);
  });

  it('refuses a download that would fill the remaining quota', () => {
    expect(hasRoomFor(base, 800)).toBe(false);
  });

  it('demands working space beyond the raw download size', () => {
    // 900MB free, so an 880MB model does not leave enough room to run.
    expect(hasRoomFor(base, 880)).toBe(false);
  });

  it('assumes room when the browser will not say', () => {
    expect(hasRoomFor({ ...base, storageQuotaMB: null, storageUsedMB: null }, 5000)).toBe(true);
  });
});

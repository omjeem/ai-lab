/**
 * Device capability detection (Section 11.3).
 *
 * WebGPU is roughly an order of magnitude faster than the WASM fallback for the
 * models here, and its absence is the single biggest driver of "why is this
 * slow". Detected once, cached, and surfaced to the user so the difference is
 * visible rather than mysterious.
 */

export type InferenceBackend = 'webgpu' | 'wasm';

export interface DeviceCapabilities {
  backend: InferenceBackend;
  webgpu: boolean;
  /** Why WebGPU was unavailable, when it was not. */
  reason: string | null;
  /** Rough storage budget, when the browser will tell us. */
  storageQuotaMB: number | null;
  storageUsedMB: number | null;
}

let cached: DeviceCapabilities | null = null;

interface GpuLike {
  requestAdapter(): Promise<unknown>;
}

export async function detectCapabilities(force = false): Promise<DeviceCapabilities> {
  if (cached && !force) return cached;

  let webgpu = false;
  let reason: string | null = null;

  if (typeof navigator === 'undefined') {
    reason = 'Not running in a browser';
  } else {
    const gpu = (navigator as Navigator & { gpu?: GpuLike }).gpu;
    if (!gpu) {
      reason = 'This browser does not expose WebGPU';
    } else {
      try {
        const adapter = await gpu.requestAdapter();
        if (adapter) {
          webgpu = true;
        } else {
          reason = 'No WebGPU adapter available on this device';
        }
      } catch (error) {
        reason = `WebGPU adapter request failed: ${(error as Error).message}`;
      }
    }
  }

  const storage = await readStorageEstimate();

  cached = {
    backend: webgpu ? 'webgpu' : 'wasm',
    webgpu,
    reason,
    ...storage,
  };
  return cached;
}

async function readStorageEstimate(): Promise<{
  storageQuotaMB: number | null;
  storageUsedMB: number | null;
}> {
  if (typeof navigator === 'undefined' || !navigator.storage?.estimate) {
    return { storageQuotaMB: null, storageUsedMB: null };
  }
  try {
    const { quota, usage } = await navigator.storage.estimate();
    return {
      storageQuotaMB: quota ? Math.round(quota / 1_000_000) : null,
      storageUsedMB: usage ? Math.round(usage / 1_000_000) : null,
    };
  } catch {
    return { storageQuotaMB: null, storageUsedMB: null };
  }
}

/** Whether a download of this size plausibly fits in the remaining quota. */
export function hasRoomFor(capabilities: DeviceCapabilities, sizeMB: number): boolean {
  if (capabilities.storageQuotaMB === null || capabilities.storageUsedMB === null) return true;
  const remaining = capabilities.storageQuotaMB - capabilities.storageUsedMB;
  // Ask for headroom beyond the raw size — the runtime needs working space too.
  return remaining > sizeMB * 1.5;
}

/** Resets the cache. Exposed for tests and for the retry path. */
export function resetCapabilitiesCache(): void {
  cached = null;
}

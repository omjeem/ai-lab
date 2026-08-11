'use client';

/**
 * Background sync for the activity queue.
 *
 * Runs on an interval and on the browser's `online` event, batches whatever is
 * queued, and clears only the ids the server confirms. Connectivity is decided
 * by two signals, since `navigator.onLine` reports "connected to something",
 * not "can reach this app" (Section 9).
 */
import { clearSynced, readQueue } from './offlineQueue';
import type { ActivityIngestResponse } from '@/types/activity';

const SYNC_INTERVAL_MS = 30_000;
const BATCH_SIZE = 100;
const PROBE_TIMEOUT_MS = 4000;

export interface SyncManagerOptions {
  onConnectivityChange?: (online: boolean) => void;
  onSynced?: (count: number) => void;
  intervalMs?: number;
}

/** True only when the browser thinks it is online *and* our API answers. */
export async function checkConnectivity(): Promise<boolean> {
  if (typeof navigator !== 'undefined' && navigator.onLine === false) return false;
  if (typeof fetch === 'undefined') return false;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), PROBE_TIMEOUT_MS);
  try {
    const response = await fetch('/api/activity', { method: 'HEAD', signal: controller.signal });
    return response.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Sends one batch. Returns how many events the server confirmed.
 *
 * Exported so a chapter can force a flush at a natural pause rather than waiting
 * out the interval.
 */
export async function syncOnce(): Promise<number> {
  const events = await readQueue(BATCH_SIZE);
  if (events.length === 0) return 0;

  const response = await fetch('/api/activity', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ events }),
  });

  if (!response.ok) return 0;

  const payload = (await response.json()) as ActivityIngestResponse;
  if (!payload.ok || !Array.isArray(payload.persistedEventIds)) return 0;

  // Only what the server acknowledged leaves the queue.
  return clearSynced(payload.persistedEventIds);
}

/** Starts the loop. Returns a teardown function for React effects. */
export function startSyncManager(options: SyncManagerOptions = {}): () => void {
  if (typeof window === 'undefined') return () => {};

  const interval = options.intervalMs ?? SYNC_INTERVAL_MS;
  let stopped = false;
  let running = false;

  const attempt = async () => {
    // Never overlap two syncs; a slow request would otherwise double-send.
    if (stopped || running) return;
    running = true;
    try {
      const online = await checkConnectivity();
      options.onConnectivityChange?.(online);
      if (!online) return;

      const synced = await syncOnce();
      if (synced > 0) options.onSynced?.(synced);
    } catch {
      options.onConnectivityChange?.(false);
    } finally {
      running = false;
    }
  };

  void attempt();
  const timer = setInterval(() => void attempt(), interval);

  const handleOnline = () => void attempt();
  const handleOffline = () => options.onConnectivityChange?.(false);
  // Coming back from a background tab is a good moment to flush.
  const handleVisibility = () => {
    if (document.visibilityState === 'visible') void attempt();
  };

  window.addEventListener('online', handleOnline);
  window.addEventListener('offline', handleOffline);
  document.addEventListener('visibilitychange', handleVisibility);

  return () => {
    stopped = true;
    clearInterval(timer);
    window.removeEventListener('online', handleOnline);
    window.removeEventListener('offline', handleOffline);
    document.removeEventListener('visibilitychange', handleVisibility);
  };
}

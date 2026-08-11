/**
 * Model lifecycle: load progress, failure state and a record of what is cached.
 *
 * transformers.js and WebLLM each cache their own weights in the browser Cache
 * API. What this module owns is everything around that: which models have
 * finished downloading (so the map can say "ready offline"), how far a download
 * has got, and the retry path when one fails (Section 11.2).
 */
import { openDB, type IDBPDatabase } from 'idb';

export type ModelLoadStatus = 'idle' | 'downloading' | 'ready' | 'error';

export interface ModelLoadState {
  modelId: string;
  status: ModelLoadStatus;
  /** 0-1, or null when the source reports no total. */
  progress: number | null;
  loadedMB: number | null;
  totalMB: number | null;
  error: string | null;
  /** Which file the loader is currently fetching, for the progress readout. */
  file: string | null;
}

export interface CachedModelRecord {
  modelId: string;
  sizeMB: number;
  cachedAt: number;
}

const DB_NAME = 'ai-learning-lab';
const DB_VERSION = 1;
const CACHE_STORE = 'modelCache';
export const ACTIVITY_STORE = 'activityQueue';
export const PROGRESS_STORE = 'gameProgress';

let dbPromise: Promise<IDBPDatabase> | null = null;

/**
 * One IndexedDB database for the whole app, per Section 7 — model records, the
 * activity queue and saved progress all live here rather than in localStorage.
 */
export function getDb(): Promise<IDBPDatabase> {
  if (typeof indexedDB === 'undefined') {
    return Promise.reject(new Error('IndexedDB is unavailable in this environment'));
  }
  dbPromise ??= openDB(DB_NAME, DB_VERSION, {
    upgrade(db) {
      if (!db.objectStoreNames.contains(CACHE_STORE)) {
        db.createObjectStore(CACHE_STORE, { keyPath: 'modelId' });
      }
      if (!db.objectStoreNames.contains(ACTIVITY_STORE)) {
        const store = db.createObjectStore(ACTIVITY_STORE, { keyPath: 'eventId' });
        store.createIndex('byTimestamp', 'timestamp');
      }
      if (!db.objectStoreNames.contains(PROGRESS_STORE)) {
        db.createObjectStore(PROGRESS_STORE);
      }
    },
  });
  return dbPromise;
}

/* ── cached-model bookkeeping ───────────────────────────────── */

export async function recordModelCached(modelId: string, sizeMB: number): Promise<void> {
  try {
    const db = await getDb();
    await db.put(CACHE_STORE, { modelId, sizeMB, cachedAt: Date.now() });
  } catch {
    // A missing IndexedDB only costs the "ready offline" badge, never gameplay.
  }
}

export async function isModelCached(modelId: string): Promise<boolean> {
  try {
    const db = await getDb();
    return (await db.get(CACHE_STORE, modelId)) !== undefined;
  } catch {
    return false;
  }
}

export async function listCachedModels(): Promise<CachedModelRecord[]> {
  try {
    const db = await getDb();
    return (await db.getAll(CACHE_STORE)) as CachedModelRecord[];
  } catch {
    return [];
  }
}

export async function forgetModel(modelId: string): Promise<void> {
  try {
    const db = await getDb();
    await db.delete(CACHE_STORE, modelId);
  } catch {
    // Nothing to clean up if the database was never reachable.
  }
}

/* ── load-progress events ───────────────────────────────────── */

type Listener = (state: ModelLoadState) => void;

const states = new Map<string, ModelLoadState>();
const listeners = new Set<Listener>();

function emit(state: ModelLoadState): void {
  states.set(state.modelId, state);
  for (const listener of listeners) listener(state);
}

export function subscribeToModelLoads(listener: Listener): () => void {
  listeners.add(listener);
  // Replay what is already known so a late subscriber is not blind.
  for (const state of states.values()) listener(state);
  return () => listeners.delete(listener);
}

export function getLoadState(modelId: string): ModelLoadState {
  return (
    states.get(modelId) ?? {
      modelId,
      status: 'idle',
      progress: null,
      loadedMB: null,
      totalMB: null,
      error: null,
      file: null,
    }
  );
}

export function markDownloading(modelId: string): void {
  emit({ ...getLoadState(modelId), status: 'downloading', error: null });
}

/** Shape transformers.js reports progress in. */
export interface RawProgress {
  status?: string;
  file?: string;
  loaded?: number;
  total?: number;
  progress?: number;
}

export function reportProgress(modelId: string, raw: RawProgress): void {
  const total = typeof raw.total === 'number' && raw.total > 0 ? raw.total : null;
  const loaded = typeof raw.loaded === 'number' ? raw.loaded : null;

  const fraction =
    typeof raw.progress === 'number'
      ? // transformers.js reports 0-100 here, WebLLM reports 0-1.
        raw.progress > 1
        ? raw.progress / 100
        : raw.progress
      : total && loaded !== null
        ? loaded / total
        : null;

  emit({
    modelId,
    status: 'downloading',
    progress: fraction === null ? null : Math.min(Math.max(fraction, 0), 1),
    loadedMB: loaded === null ? null : Math.round(loaded / 1_000_000),
    totalMB: total === null ? null : Math.round(total / 1_000_000),
    error: null,
    file: raw.file ?? null,
  });
}

export function markReady(modelId: string): void {
  emit({ ...getLoadState(modelId), status: 'ready', progress: 1, error: null });
}

export function markError(modelId: string, error: unknown): void {
  emit({
    ...getLoadState(modelId),
    status: 'error',
    error: error instanceof Error ? error.message : String(error),
  });
}

/** Clears the error so the retry button can start a fresh attempt. */
export function clearError(modelId: string): void {
  emit({ ...getLoadState(modelId), status: 'idle', error: null, progress: null });
}

/** Test seam — drops all in-memory load state. */
export function resetLoadStates(): void {
  states.clear();
  listeners.clear();
}

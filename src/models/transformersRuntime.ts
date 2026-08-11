/**
 * Shared transformers.js plumbing: one configured runtime, one place that
 * decides WebGPU versus WASM, and one loader that reports progress and records
 * a successful download.
 *
 * Every wrapper in this folder goes through `loadOnce`, so retry, caching and
 * progress behave identically no matter which model failed.
 */
import { detectCapabilities, type InferenceBackend } from '@/lib/deviceCapabilities';
import {
  markDownloading,
  markError,
  markReady,
  recordModelCached,
  reportProgress,
  type RawProgress,
} from './modelCache';

export type TransformersModule = typeof import('@huggingface/transformers');

let modulePromise: Promise<TransformersModule> | null = null;

/**
 * Imports transformers.js lazily.
 *
 * It is a large dependency that touches browser APIs on import, so it must stay
 * out of the initial bundle and out of any server render.
 */
export async function getTransformers(): Promise<TransformersModule> {
  modulePromise ??= import('@huggingface/transformers').then((mod) => {
    // Models come from the Hub and are cached by the browser, which is what
    // makes every browser-tier chapter work offline after the first visit.
    mod.env.allowLocalModels = false;
    mod.env.useBrowserCache = true;
    return mod;
  });
  return modulePromise;
}

export interface LoadOptions {
  modelId: string;
  estimatedSizeMB?: number;
  /** Forces a backend; otherwise detected once per session. */
  backend?: InferenceBackend;
}

const instances = new Map<string, Promise<unknown>>();

/**
 * Loads a model exactly once per session, wiring progress and failure state.
 *
 * A failed load is evicted so the retry button can genuinely try again rather
 * than resolving the same rejected promise forever.
 */
export async function loadOnce<T>(
  key: string,
  options: LoadOptions,
  factory: (context: {
    transformers: TransformersModule;
    backend: InferenceBackend;
    onProgress: (raw: RawProgress) => void;
  }) => Promise<T>
): Promise<T> {
  const existing = instances.get(key);
  if (existing) return existing as Promise<T>;

  const promise = (async () => {
    markDownloading(options.modelId);
    try {
      const transformers = await getTransformers();
      const backend = options.backend ?? (await detectCapabilities()).backend;

      const value = await factory({
        transformers,
        backend,
        onProgress: (raw) => reportProgress(options.modelId, raw),
      });

      markReady(options.modelId);
      void recordModelCached(options.modelId, options.estimatedSizeMB ?? 0);
      return value;
    } catch (error) {
      markError(options.modelId, error);
      // Drop the rejected promise so a retry is a real second attempt.
      instances.delete(key);
      throw error;
    }
  })();

  instances.set(key, promise);
  return promise;
}

/** Discards a loaded model so the next call re-creates it. */
export function evictModel(key: string): void {
  instances.delete(key);
}

export function evictAllModels(): void {
  instances.clear();
}

/** Converts a transformers.js tensor-ish object into plain nested arrays. */
export function toNestedArray(tensor: unknown): number[] {
  const candidate = tensor as { tolist?: () => unknown; data?: ArrayLike<number> };
  if (typeof candidate?.tolist === 'function') {
    return candidate.tolist() as number[];
  }
  if (candidate?.data) return Array.from(candidate.data);
  return [];
}

/** Numerically stable softmax over a plain array. */
export function softmaxRow(logits: readonly number[]): number[] {
  if (logits.length === 0) return [];
  const max = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - max));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / total);
}

export interface RankedToken {
  token: string;
  id: number;
  probability: number;
}

/** Top-k entries of a distribution, highest probability first. */
export function topK(
  probabilities: readonly number[],
  k: number,
  decode: (id: number) => string
): RankedToken[] {
  return probabilities
    .map((probability, id) => ({ id, probability }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, Math.max(1, k))
    .map(({ id, probability }) => ({ id, probability, token: decode(id) }));
}

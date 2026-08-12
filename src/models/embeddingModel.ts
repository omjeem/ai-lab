/**
 * Real sentence embeddings from all-MiniLM-L6-v2, running in the browser.
 *
 * Implements the `Embedder` interface the World 1 and 5.1 engines consume, so
 * the same code path that tests exercise with a fake runs against the real model
 * in the app.
 */
import type { Embedder } from '@/engines/deps';
import { loadOnce, toNestedArray, type TransformersModule } from './transformersRuntime';

export const EMBEDDING_MODEL_ID = 'Xenova/all-MiniLM-L6-v2';
const EMBEDDING_SIZE_MB = 23;

type FeatureExtractor = Awaited<ReturnType<TransformersModule['pipeline']>>;

/**
 * The loaded pipeline itself.
 *
 * Exposed so the vocabulary index can read this model's own tokenizer rather
 * than shipping a word list beside it.
 */
export async function getEmbeddingExtractor(): Promise<FeatureExtractor> {
  return getExtractor();
}

async function getExtractor(): Promise<FeatureExtractor> {
  return loadOnce<FeatureExtractor>(
    'embedding',
    { modelId: EMBEDDING_MODEL_ID, estimatedSizeMB: EMBEDDING_SIZE_MB },
    async ({ transformers, backend, onProgress }) =>
      transformers.pipeline('feature-extraction', EMBEDDING_MODEL_ID, {
        device: backend,
        progress_callback: onProgress,
      })
  );
}

async function meanPooled(texts: string[], normalize: boolean): Promise<number[][]> {
  if (texts.length === 0) return [];

  const extractor = await getExtractor();
  const output = await (extractor as unknown as (
    input: string[],
    options: { pooling: 'mean'; normalize: boolean }
  ) => Promise<unknown>)(texts, { pooling: 'mean', normalize });

  const nested = toNestedArray(output) as unknown;

  // A single input can come back un-nested depending on the pipeline version.
  if (Array.isArray(nested) && typeof nested[0] === 'number') {
    return [nested as number[]];
  }

  const rows = nested as number[][];
  if (!Array.isArray(rows) || rows.length !== texts.length) {
    throw new Error(
      `Embedding model returned ${Array.isArray(rows) ? rows.length : 0} vectors for ${texts.length} inputs`
    );
  }
  return rows;
}

/**
 * Mean-pooled, L2-normalised embeddings — the standard way this model is used,
 * and what makes cosine similarity the right comparison for World 1.3.
 */
export const embeddingModel: Embedder = {
  embed: (texts) => meanPooled(texts, true),
};

/**
 * The same embeddings with their magnitudes left intact.
 *
 * Needed wherever length carries meaning. On unit vectors Euclidean distance is
 * `sqrt(2 - 2cos)`, a strictly decreasing function of cosine, so the two metrics
 * can never order anything differently — measured against this model, 0 of 336
 * triples disagree normalised against 74 of 336 raw. Chapter 1.3's third level
 * asks the player to find that disagreement, which only exists here.
 */
export const rawEmbeddingModel: Embedder = {
  embed: (texts) => meanPooled(texts, false),
};

/** Warms the model so a chapter can show its download progress up front. */
export async function preloadEmbeddingModel(): Promise<void> {
  await getExtractor();
}

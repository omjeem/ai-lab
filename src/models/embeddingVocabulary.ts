/**
 * The embedding model's own vocabulary, as a searchable index.
 *
 * A candidate pool written into a level's JSON can only ever return words
 * somebody chose in advance, which makes the arithmetic look staged: compute
 * `animal + cats + dog` against a pool of countries and the instrument still
 * answers "germany". The fix is not a bigger hand-written list — it is to search
 * the vocabulary the model already has.
 *
 * The word list here is read out of the loaded tokenizer at runtime. Nothing is
 * authored: WordPiece continuations (`##ing`), punctuation and single letters
 * are dropped, and what remains is taken in token-id order, which for this
 * tokenizer is corpus frequency order. Every word is then embedded by the same
 * model the chapter is already running, so a neighbour is a real neighbour.
 *
 * The index is display-only. Scoring stays with the engine and the pass criteria
 * in the level JSON, which are calibrated against their own pools.
 */
import { embeddingModel, getEmbeddingExtractor } from './embeddingModel';

/**
 * How many words to index.
 *
 * Frequency-ordered, so 8000 reaches words like "cats" (5789) and "swimming"
 * (3164) while staying near 12MB of Float32 and a few seconds to embed.
 */
export const DEFAULT_VOCABULARY_SIZE = 8000;

/** Batched so a slow device reports progress instead of freezing. */
const EMBED_CHUNK = 256;

/** Whole lowercase words only — no `##` continuations, digits or punctuation. */
const WHOLE_WORD = /^[a-z]{3,}$/;

export interface VocabularyIndex {
  words: string[];
  /** Row-major, one L2-normalised row per word. */
  matrix: Float32Array;
  dims: number;
}

export interface VocabularyNeighbour {
  word: string;
  similarity: number;
}

/**
 * Picks the indexable words out of a raw `{token: id}` vocabulary.
 *
 * Token id ascending is the model's own frequency order, so slicing the front
 * gives common English rather than an arbitrary sample.
 */
export function selectVocabularyWords(
  vocab: Record<string, number>,
  limit: number
): string[] {
  return Object.entries(vocab)
    .filter(([token]) => WHOLE_WORD.test(token))
    .sort((a, b) => a[1] - b[1])
    .slice(0, Math.max(0, limit))
    .map(([token]) => token);
}

/**
 * Nearest words to a vector, highest cosine similarity first.
 *
 * Rows are already unit length, so the dot product is the cosine once the query
 * is divided by its own norm. Top-k is kept by insertion because k is small and
 * the alternative is allocating an object per vocabulary entry on every keystroke.
 */
export function nearestInIndex(
  index: VocabularyIndex,
  query: readonly number[],
  k: number,
  exclude?: ReadonlySet<string>
): VocabularyNeighbour[] {
  const { words, matrix, dims } = index;
  if (words.length === 0 || dims === 0 || query.length !== dims || k <= 0) return [];

  let queryNorm = 0;
  for (let i = 0; i < dims; i++) queryNorm += query[i]! * query[i]!;
  queryNorm = Math.sqrt(queryNorm);
  if (queryNorm === 0) return [];

  const best: VocabularyNeighbour[] = [];
  for (let row = 0; row < words.length; row++) {
    const word = words[row]!;
    if (exclude?.has(word)) continue;

    const offset = row * dims;
    let dot = 0;
    for (let i = 0; i < dims; i++) dot += matrix[offset + i]! * query[i]!;
    const similarity = dot / queryNorm;

    if (best.length === k && similarity <= best[best.length - 1]!.similarity) continue;

    let position = best.length;
    while (position > 0 && best[position - 1]!.similarity < similarity) position--;
    best.splice(position, 0, { word, similarity });
    if (best.length > k) best.pop();
  }
  return best;
}

/** Reads the raw vocabulary off the loaded tokenizer, accepting either shape. */
async function readVocabulary(): Promise<Record<string, number>> {
  const extractor = (await getEmbeddingExtractor()) as unknown as {
    tokenizer?: {
      _tokenizerJSON?: { model?: { vocab?: unknown } };
      model?: { vocab?: unknown };
    };
  };

  const tokenizer = extractor.tokenizer;
  // The field has moved between transformers.js versions; accept both rather
  // than failing the panel over a rename.
  const raw = tokenizer?._tokenizerJSON?.model?.vocab ?? tokenizer?.model?.vocab;

  if (Array.isArray(raw)) {
    const byToken: Record<string, number> = {};
    raw.forEach((token, id) => {
      if (typeof token === 'string') byToken[token] = id;
    });
    return byToken;
  }
  if (raw && typeof raw === 'object') return raw as Record<string, number>;

  throw new Error("Could not read the embedding model's vocabulary from its tokenizer");
}

let cached: { size: number; index: VocabularyIndex } | null = null;
let inflight: { size: number; promise: Promise<VocabularyIndex> } | null = null;

/**
 * Builds (or returns) the index for a vocabulary size.
 *
 * Held in memory for the session rather than persisted — the weights are already
 * cached by the browser, and re-embedding costs seconds, not a download.
 */
export async function loadVocabularyIndex(
  size: number = DEFAULT_VOCABULARY_SIZE,
  onProgress?: (done: number, total: number) => void
): Promise<VocabularyIndex> {
  if (cached?.size === size) {
    onProgress?.(cached.index.words.length, cached.index.words.length);
    return cached.index;
  }
  if (inflight?.size === size) return inflight.promise;

  const promise = (async (): Promise<VocabularyIndex> => {
    const words = selectVocabularyWords(await readVocabulary(), size);
    if (words.length === 0) {
      throw new Error('The tokenizer vocabulary held no indexable words');
    }

    let matrix: Float32Array | null = null;
    let dims = 0;

    for (let start = 0; start < words.length; start += EMBED_CHUNK) {
      const chunk = words.slice(start, start + EMBED_CHUNK);
      const vectors = await embeddingModel.embed(chunk);

      if (matrix === null) {
        dims = vectors[0]?.length ?? 0;
        if (dims === 0) throw new Error('The embedding model returned empty vectors');
        matrix = new Float32Array(words.length * dims);
      }
      vectors.forEach((vector, i) => matrix!.set(vector, (start + i) * dims));
      onProgress?.(Math.min(start + chunk.length, words.length), words.length);
    }

    const index: VocabularyIndex = { words, matrix: matrix!, dims };
    cached = { size, index };
    return index;
  })();

  inflight = { size, promise };
  try {
    return await promise;
  } finally {
    if (inflight?.promise === promise) inflight = null;
  }
}

/**
 * The real GPT-2 byte-pair tokenizer.
 *
 * World 1.4 scores merge order against this tokenizer's own merge table, so the
 * table is read out of the loaded tokenizer rather than reconstructed.
 */
import type { TokenizerDep } from '@/engines/deps';
import { loadOnce, type TransformersModule } from './transformersRuntime';

export const TOKENIZER_MODEL_ID = 'Xenova/gpt2';
const TOKENIZER_SIZE_MB = 4;

type Tokenizer = Awaited<ReturnType<TransformersModule['AutoTokenizer']['from_pretrained']>>;

async function getTokenizer(): Promise<Tokenizer> {
  return loadOnce<Tokenizer>(
    'tokenizer',
    { modelId: TOKENIZER_MODEL_ID, estimatedSizeMB: TOKENIZER_SIZE_MB },
    async ({ transformers, onProgress }) =>
      transformers.AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID, {
        progress_callback: onProgress,
      })
  );
}

/** GPT-2 encodes bytes as printable characters; undo that for display. */
export function prettifyToken(token: string): string {
  return token.replace(/Ġ/g, ' ').replace(/Ċ/g, '\n');
}

let mergeRanksCache: ReadonlyMap<string, number> | null = null;

/**
 * Reads the tokenizer's own BPE merge list.
 *
 * The merges live on the loaded tokenizer's model; the exact field has moved
 * between transformers.js versions, so several shapes are accepted rather than
 * failing the chapter over a rename.
 */
async function readMergeRanks(): Promise<ReadonlyMap<string, number>> {
  if (mergeRanksCache) return mergeRanksCache;

  const tokenizer = (await getTokenizer()) as unknown as {
    model?: { merges?: unknown; bpe_ranks?: Map<string, number> };
  };

  const ranks = new Map<string, number>();
  const model = tokenizer.model;

  if (model?.bpe_ranks instanceof Map) {
    for (const [pair, rank] of model.bpe_ranks) ranks.set(pair, rank);
  } else if (Array.isArray(model?.merges)) {
    model.merges.forEach((entry: unknown, index: number) => {
      // Older builds store "a b"; newer ones store ["a", "b"].
      const key = Array.isArray(entry) ? entry.join(' ') : String(entry);
      if (!ranks.has(key)) ranks.set(key, index);
    });
  }

  if (ranks.size === 0) {
    throw new Error('Could not read the BPE merge table from the loaded tokenizer');
  }

  mergeRanksCache = ranks;
  return ranks;
}

export const tokenizerModel: TokenizerDep = {
  async tokenize(text: string): Promise<string[]> {
    const tokenizer = (await getTokenizer()) as unknown as {
      tokenize?: (t: string) => string[];
      encode: (t: string) => number[];
    };
    if (typeof tokenizer.tokenize === 'function') return tokenizer.tokenize(text);

    // Fall back to decoding ids one by one if `tokenize` is unavailable.
    const ids = tokenizer.encode(text);
    const decoded = await Promise.all(ids.map((id) => tokenizerModel.decode([id])));
    return decoded;
  },

  async encode(text: string): Promise<number[]> {
    const tokenizer = (await getTokenizer()) as unknown as { encode: (t: string) => number[] };
    return tokenizer.encode(text);
  },

  async decode(ids: number[]): Promise<string> {
    const tokenizer = (await getTokenizer()) as unknown as {
      decode: (ids: number[], options?: Record<string, unknown>) => string;
    };
    return tokenizer.decode(ids, { skip_special_tokens: false });
  },

  mergeRanks: readMergeRanks,
};

export async function preloadTokenizer(): Promise<void> {
  await getTokenizer();
}

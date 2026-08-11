/**
 * Loads the bundled public-domain corpora the n-gram and RNN chapters count
 * from. Fetched at runtime and cached in memory, so World 4 builds its tables
 * from text rather than shipping precomputed statistics.
 */
import type { CorpusDep } from '@/engines/deps';

const CORPUS_PATHS: Record<string, string> = {
  'austen-sample': '/corpora/austen-sample.txt',
};

const cache = new Map<string, string>();

export const corpusLoader: CorpusDep = {
  async load(corpusId: string): Promise<string> {
    const cached = cache.get(corpusId);
    if (cached !== undefined) return cached;

    const path = CORPUS_PATHS[corpusId];
    if (!path) throw new Error(`Unknown corpus "${corpusId}"`);

    const response = await fetch(path);
    if (!response.ok) {
      throw new Error(`Could not load corpus "${corpusId}" (${response.status})`);
    }

    const text = await response.text();
    if (text.trim().length === 0) throw new Error(`Corpus "${corpusId}" is empty`);

    cache.set(corpusId, text);
    return text;
  },
};

export function availableCorpora(): string[] {
  return Object.keys(CORPUS_PATHS);
}

export function clearCorpusCache(): void {
  cache.clear();
}

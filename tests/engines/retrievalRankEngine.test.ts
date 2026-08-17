import { describe, it, expect } from 'vitest';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  type RetrievalRankConfig,
} from '@/engines/retrievalRankEngine';
import type { CorpusDep, Embedder } from '@/engines/deps';
import type { EngineRules } from '@/types/game';

/**
 * Synthetic 2D fixture — chosen so every cosine similarity is hand-computable,
 * not because it resembles real embeddings. `a`/`c` and `b`/`d` are paired into
 * two documents; each fact's three sentences share its passage vector so
 * "which fact does this chunk belong to" is unambiguous in assertions.
 */
const FACT_VECTORS: Record<string, number[]> = {
  a: [1, 0],
  b: [0, 1],
  c: [-1, 0],
  d: [0, -1],
};

const CORPUS = {
  facts: [
    {
      id: 'a',
      topic: 'Alpha',
      sentences: ['Alpha sentence one.', 'Alpha sentence two with 42.', 'Alpha sentence three.'],
      query: 'What is alpha?',
      answer: '42',
    },
    {
      id: 'b',
      topic: 'Beta',
      sentences: ['Beta sentence one.', 'Beta sentence two with 7.', 'Beta sentence three.'],
      query: 'What is beta?',
      answer: '7',
    },
    {
      id: 'c',
      topic: 'Gamma',
      sentences: ['Gamma sentence one.', 'Gamma sentence two with 99.', 'Gamma sentence three.'],
      query: 'What is gamma?',
      answer: '99',
    },
    {
      id: 'd',
      topic: 'Delta',
      sentences: ['Delta sentence one.', 'Delta sentence two with 5.', 'Delta sentence three.'],
      query: 'What is delta?',
      answer: '5',
    },
  ],
  documents: [
    { id: 'doc-ac', factIds: ['a', 'c'] },
    { id: 'doc-bd', factIds: ['b', 'd'] },
  ],
};

const RANK_QUERY = 'Which is most alpha-like?';
// a=0.8, b=0.6, d=-0.6, c=-0.8 against [0.8, 0.6] — a strict order, no ties.
const RANK_QUERY_VECTOR = [0.8, 0.6];

const DOC_VECTORS: Record<string, number[]> = {
  'doc-ac': [0.9, 0],
  'doc-bd': [0, 0.9],
};

/** Adversarial only for fact "d" — deliberately closer to "a" than to "d" itself. */
const FACT_QUERY_VECTORS: Record<string, number[]> = {
  a: [1, 0],
  b: [0, 1],
  c: [-1, 0],
  d: [0.9, -0.1],
};

function vectorFor(text: string): number[] {
  if (text === RANK_QUERY) return RANK_QUERY_VECTOR;
  for (const doc of CORPUS.documents) {
    const full = doc.factIds.flatMap((fid) => CORPUS.facts.find((f) => f.id === fid)!.sentences).join(' ');
    if (text === full) return DOC_VECTORS[doc.id]!;
  }
  for (const fact of CORPUS.facts) {
    const passage = fact.sentences.join(' ');
    if (text === passage) return FACT_VECTORS[fact.id]!;
    if (text === fact.query) return FACT_QUERY_VECTORS[fact.id]!;
    if (fact.sentences.includes(text)) return FACT_VECTORS[fact.id]!;
  }
  throw new Error(`fakeEmbedder: no fixture vector for ${JSON.stringify(text)}`);
}

const fakeEmbedder: Embedder = {
  embed: async (texts) => texts.map(vectorFor),
};

const fakeCorpus: CorpusDep = {
  async load(id: string) {
    if (id === 'bad-corpus') {
      return JSON.stringify({
        facts: [{ ...CORPUS.facts[0], answer: 'not-in-passage-anywhere' }],
        documents: [],
      });
    }
    return JSON.stringify(CORPUS);
  },
};

const rankRules: EngineRules = {
  passCriteria: { metric: 'rankCorrelation', threshold: 0.5, comparator: 'gte' },
  starsRules: [
    { threshold: 0.5, stars: 1 },
    { threshold: 0.8, stars: 2 },
    { threshold: 0.99, stars: 3 },
  ],
  xpReward: 40,
};

const breakRules: EngineRules = {
  passCriteria: { metric: 'retrievalMargin', threshold: 0, comparator: 'gte' },
  starsRules: [
    { threshold: 0, stars: 1 },
    { threshold: 0.2, stars: 2 },
    { threshold: 0.4, stars: 3 },
  ],
  xpReward: 45,
};

const chunkRules: EngineRules = {
  passCriteria: { metric: 'retrievalPrecision', threshold: 0.5, comparator: 'gte' },
  starsRules: [
    { threshold: 0.5, stars: 1 },
    { threshold: 0.75, stars: 2 },
    { threshold: 1, stars: 3 },
  ],
  xpReward: 50,
};

describe('retrievalRankEngine — prepare', () => {
  it('loads facts and documents and builds every vector', async () => {
    const config: RetrievalRankConfig = { mode: 'rank', corpus: 'retrieval-facts', query: RANK_QUERY };
    const prepared = await prepare(config, { corpus: fakeCorpus, embedder: fakeEmbedder });

    expect(prepared.facts.map((f) => f.id).sort()).toEqual(['a', 'b', 'c', 'd']);
    expect(prepared.documents.map((d) => d.id).sort()).toEqual(['doc-ac', 'doc-bd']);
    expect(prepared.factVectors.a).toEqual([1, 0]);
    expect(prepared.sentenceVectors.a).toHaveLength(3);
    expect(prepared.documentVectors['doc-ac']).toEqual([0.9, 0]);
    expect(prepared.factQueryVectors.a).toEqual([1, 0]);
    expect(prepared.rankQueryVector).toEqual(RANK_QUERY_VECTOR);
  });

  it('rejects a fact whose answer never appears in its own passage', async () => {
    const config: RetrievalRankConfig = { mode: 'rank', corpus: 'bad-corpus', query: RANK_QUERY };
    await expect(prepare(config, { corpus: fakeCorpus, embedder: fakeEmbedder })).rejects.toThrow();
  });

  it('leaves rankQueryVector null outside rank mode', async () => {
    const config: RetrievalRankConfig = { mode: 'break-retriever', corpus: 'retrieval-facts', targetFactId: 'c' };
    const prepared = await prepare(config, { corpus: fakeCorpus, embedder: fakeEmbedder });
    expect(prepared.rankQueryVector).toBeNull();
  });
});

describe('retrievalRankEngine — rank mode', () => {
  const config: RetrievalRankConfig = {
    mode: 'rank',
    corpus: 'retrieval-facts',
    query: RANK_QUERY,
    factIds: ['a', 'b', 'c', 'd'],
  };

  it('computes the true order from real cosine similarity to the query', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, embedder: fakeEmbedder });
    const state = initState(config, rankRules, prepared);
    expect(state.trueOrder).toEqual(['a', 'b', 'd', 'c']);
  });

  it('scores a perfect match as full correlation and full stars', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, embedder: fakeEmbedder });
    let state = initState(config, rankRules, prepared);
    state = applyAction(state, { type: 'SET_ORDER', ordering: state.trueOrder });

    const result = evaluate(state);
    expect(result.metric).toBe('rankCorrelation');
    expect(result.value).toBeCloseTo(1);
    expect(result.passed).toBe(true);
    expect(result.stars).toBe(3);
  });

  it('scores a fully reversed order as -1 and a fail', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, embedder: fakeEmbedder });
    let state = initState(config, rankRules, prepared);
    state = applyAction(state, { type: 'SET_ORDER', ordering: [...state.trueOrder].reverse() });

    const result = evaluate(state);
    expect(result.value).toBeCloseTo(-1);
    expect(result.passed).toBe(false);
    expect(result.stars).toBe(0);
  });

  it('rejects a reorder that drops or duplicates an item', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, embedder: fakeEmbedder });
    const state = initState(config, rankRules, prepared);
    const next = applyAction(state, { type: 'SET_ORDER', ordering: ['a', 'a', 'b', 'c'] });
    expect(next).toBe(state);
  });

  it('MOVE_ITEM relocates a single entry', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, embedder: fakeEmbedder });
    let state = initState(config, rankRules, prepared);
    state = applyAction(state, { type: 'MOVE_ITEM', from: 0, to: 3 });
    expect(state.ordering).toEqual(['b', 'c', 'd', 'a']);
  });
});

describe('retrievalRankEngine — break-retriever mode', () => {
  const config: RetrievalRankConfig = { mode: 'break-retriever', corpus: 'retrieval-facts', targetFactId: 'c' };

  it('fails with no submitted query', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, embedder: fakeEmbedder });
    const state = initState(config, breakRules, prepared);
    const result = evaluate(state);
    expect(result.passed).toBe(false);
  });

  it('scores a positive margin when the target genuinely ranks first', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, embedder: fakeEmbedder });
    let state = initState(config, breakRules, prepared);
    // Matches "c" exactly: sims a=-1, b=0, c=1, d=0 -> margin = 1 - 0 = 1.
    state = applyAction(state, { type: 'SUBMIT_QUERY', query: 'gamma', vector: [-1, 0] });

    const result = evaluate(state);
    expect(result.metric).toBe('retrievalMargin');
    expect(result.value).toBeCloseTo(1);
    expect(result.passed).toBe(true);
    expect(result.stars).toBe(3);
  });

  it('fails when the query retrieves a different passage instead', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, embedder: fakeEmbedder });
    let state = initState(config, breakRules, prepared);
    // Matches "a": sims a=1, b=0, c=-1, d=0 -> target margin = -1 - 1 = -2.
    state = applyAction(state, { type: 'SUBMIT_QUERY', query: 'alpha', vector: [1, 0] });

    const result = evaluate(state);
    expect(result.value).toBeCloseTo(-2);
    expect(result.passed).toBe(false);
  });
});

describe('retrievalRankEngine — chunking mode', () => {
  const config: RetrievalRankConfig = {
    mode: 'chunking',
    corpus: 'retrieval-facts',
    documentIds: ['doc-ac', 'doc-bd'],
  };

  it('builds one round per fact across the configured documents', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, embedder: fakeEmbedder });
    const state = initState(config, chunkRules, prepared);
    expect(state.chunkRounds.map((r) => r.factId).sort()).toEqual(['a', 'b', 'c', 'd']);
    expect(state.chunkRounds.every((r) => r.strategy === null)).toBe(true);
  });

  it('scores 0 precision for an unanswered round', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, embedder: fakeEmbedder });
    const state = initState(config, chunkRules, prepared);
    const result = evaluate(state);
    expect(result.value).toBe(0);
  });

  it('paragraph strategy scores 0.5 precision when the doc is correctly identified', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, embedder: fakeEmbedder });
    let state = initState(config, chunkRules, prepared);
    const roundIndex = state.chunkRounds.findIndex((r) => r.factId === 'a');
    state = applyAction(state, { type: 'SET_STRATEGY', roundIndex, strategy: 'paragraph' });

    const result = evaluate(state);
    // Only this round answered: (0.5 + 0 + 0 + 0) / 4 = 0.125.
    expect(result.value).toBeCloseTo(0.125);
  });

  it('sentence strategy scores full precision when the exact chunk is found', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, embedder: fakeEmbedder });
    let state = initState(config, chunkRules, prepared);
    const roundIndex = state.chunkRounds.findIndex((r) => r.factId === 'a');
    state = applyAction(state, { type: 'SET_STRATEGY', roundIndex, strategy: 'sentence' });

    const result = evaluate(state);
    expect(result.value).toBeCloseTo(0.25); // (1.0 + 0 + 0 + 0) / 4
  });

  it('scores 0 for both strategies when the query is adversarially misleading', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, embedder: fakeEmbedder });
    const roundIndex = 0;
    let paraState = initState(config, chunkRules, prepared);
    const dIndexP = paraState.chunkRounds.findIndex((r) => r.factId === 'd');
    paraState = applyAction(paraState, { type: 'SET_STRATEGY', roundIndex: dIndexP, strategy: 'paragraph' });
    expect(evaluate(paraState).value).toBeCloseTo(0);

    let sentState = initState(config, chunkRules, prepared);
    const dIndexS = sentState.chunkRounds.findIndex((r) => r.factId === 'd');
    sentState = applyAction(sentState, { type: 'SET_STRATEGY', roundIndex: dIndexS, strategy: 'sentence' });
    expect(evaluate(sentState).value).toBeCloseTo(0);
    void roundIndex;
  });

  it('averages precision across every answered round', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus, embedder: fakeEmbedder });
    let state = initState(config, chunkRules, prepared);
    const aIndex = state.chunkRounds.findIndex((r) => r.factId === 'a');
    const bIndex = state.chunkRounds.findIndex((r) => r.factId === 'b');
    state = applyAction(state, { type: 'SET_STRATEGY', roundIndex: aIndex, strategy: 'sentence' });
    state = applyAction(state, { type: 'SET_STRATEGY', roundIndex: bIndex, strategy: 'sentence' });

    const result = evaluate(state);
    // a and b both correct sentence retrieval (1.0 each), c and d unanswered (0 each): 2/4 = 0.5.
    expect(result.value).toBeCloseTo(0.5);
    expect(result.passed).toBe(true);
    expect(result.stars).toBe(1);
  });
});

describe('retrievalRankEngine — reset and submit', () => {
  it('RESET restores the original ordering without re-fetching data', async () => {
    const config: RetrievalRankConfig = {
      mode: 'rank',
      corpus: 'retrieval-facts',
      query: RANK_QUERY,
      factIds: ['a', 'b', 'c', 'd'],
    };
    const prepared = await prepare(config, { corpus: fakeCorpus, embedder: fakeEmbedder });
    let state = initState(config, rankRules, prepared);
    const original = [...state.ordering];
    state = applyAction(state, { type: 'MOVE_ITEM', from: 0, to: 2 });
    expect(state.ordering).not.toEqual(original);

    state = applyAction(state, { type: 'RESET' });
    expect(state.ordering).toEqual(original);
    expect(state.status).toBe('idle');
  });

  it('SUBMIT marks the run complete', async () => {
    const config: RetrievalRankConfig = { mode: 'break-retriever', corpus: 'retrieval-facts', targetFactId: 'c' };
    const prepared = await prepare(config, { corpus: fakeCorpus, embedder: fakeEmbedder });
    let state = initState(config, breakRules, prepared);
    state = applyAction(state, { type: 'SUBMIT' });
    expect(state.status).toBe('complete');
  });
});

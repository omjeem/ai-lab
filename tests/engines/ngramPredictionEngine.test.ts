import { describe, it, expect } from 'vitest';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  tokenize,
  buildTable,
  probabilityOf,
  heldOutPerplexity,
  contextCoverage,
  topPredictions,
  type NgramPredictionConfig,
} from '@/engines/ngramPredictionEngine';
import type { CorpusDep } from '@/engines/deps';
import type { EngineRules } from '@/types/game';
import game from '@data/games/world-4-sequence-models/4-1-ngrams.json';

/**
 * Corpus with planted bigram statistics — "the cat" is by far the most common
 * continuation of "the" — but non-repeating sentence structure, so long
 * contexts genuinely go unseen. A verbatim-repeated corpus would keep coverage
 * at 1 for every order and hide the sparsity wall the chapter is about.
 */
const CORPUS = (() => {
  let seed = 4242;
  const rand = (n: number) => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    // Take the high bits — an LCG's low-order bits cycle with a tiny period,
    // which would make the "random" corpus repeat and hide the sparsity wall.
    return Math.floor((seed / 0x80000000) * n);
  };
  const verbs = ['sat', 'ate', 'slept', 'watched', 'chased', 'ignored', 'found', 'left'];
  const places = ['mat', 'fish', 'garden', 'window', 'chair', 'bird', 'road', 'box'];
  const openers = ['then', 'later', 'quietly', 'again', 'meanwhile', 'once', 'briefly'];

  const sentences: string[] = [];
  for (let i = 0; i < 400; i++) {
    const subject = rand(10) === 0 ? 'the dog' : 'the cat';
    sentences.push(
      `${openers[rand(openers.length)]} ${subject} ${verbs[rand(verbs.length)]} ` +
        `near the ${places[rand(places.length)]}.`
    );
  }
  return sentences.join(' ');
})();

const fakeCorpus: CorpusDep = {
  async load(id: string) {
    if (id === 'empty') return '';
    return CORPUS;
  },
};

const rulesFor = (i: number): EngineRules => {
  const level = game.levels[i]!;
  return {
    passCriteria: level.passCriteria as EngineRules['passCriteria'],
    starsRules: level.starsRules,
    xpReward: level.xpReward,
  };
};
const configFor = (i: number) => game.levels[i]!.engineConfig as unknown as NgramPredictionConfig;

describe('tokenize', () => {
  it('lower-cases and separates punctuation', () => {
    expect(tokenize('The Cat, sat.')).toEqual(['the', 'cat', ',', 'sat', '.']);
  });

  it('keeps apostrophes inside words', () => {
    expect(tokenize("don't")).toEqual(["don't"]);
  });

  it('returns an empty list for empty text', () => {
    expect(tokenize('')).toEqual([]);
  });
});

describe('n-gram tables', () => {
  const tokens = tokenize('a b a b a c');

  it('counts bigram continuations from the tokens', () => {
    const table = buildTable(tokens, 2, 3);
    expect(table.counts.get('a')!.get('b')).toBe(2);
    expect(table.counts.get('a')!.get('c')).toBe(1);
    expect(table.contextTotals.get('a')).toBe(3);
  });

  it('treats a unigram table as having a single empty context', () => {
    const table = buildTable(tokens, 1, 3);
    expect(table.counts.size).toBe(1);
    expect(table.contextTotals.get('')).toBe(tokens.length);
  });

  it('produces probabilities that sum to one over the vocabulary', () => {
    const table = buildTable(tokens, 2, 3);
    const vocabulary = ['a', 'b', 'c'];
    const total = vocabulary.reduce((sum, w) => sum + probabilityOf(table, 'a', w, 0.1), 0);
    expect(total).toBeCloseTo(1);
  });

  it('gives unseen continuations non-zero probability under smoothing', () => {
    const table = buildTable(tokens, 2, 3);
    expect(probabilityOf(table, 'a', 'zzz', 0)).toBe(0);
    expect(probabilityOf(table, 'a', 'zzz', 0.1)).toBeGreaterThan(0);
  });

  it('falls back to uniform for a context it has never seen', () => {
    const table = buildTable(tokens, 2, 3);
    expect(probabilityOf(table, 'unseen-context', 'a', 0)).toBeCloseTo(1 / 3);
  });
});

describe('perplexity and coverage', () => {
  it('reports low perplexity on text the table has memorised', () => {
    const tokens = tokenize('a b a b a b a b a b');
    const table = buildTable(tokens, 2, 2);
    expect(heldOutPerplexity(table, tokens, 0.001)).toBeLessThan(2);
  });

  it('reports higher perplexity on unfamiliar text', () => {
    const table = buildTable(tokenize('a b a b a b'), 2, 4);
    const familiar = heldOutPerplexity(table, tokenize('a b a b'), 0.1);
    const strange = heldOutPerplexity(table, tokenize('c d c d'), 0.1);
    expect(strange).toBeGreaterThan(familiar);
  });

  it('reports full coverage for contexts it has all seen', () => {
    const tokens = tokenize('a b a b a b');
    const table = buildTable(tokens, 2, 2);
    expect(contextCoverage(table, tokens)).toBe(1);
  });

  it('reports zero coverage for entirely unseen contexts', () => {
    const table = buildTable(tokenize('a b a b'), 2, 4);
    expect(contextCoverage(table, tokenize('x y x y'))).toBe(0);
  });
});

describe('ngramPredictionEngine — prepare and setup', () => {
  it('tokenizes the loaded corpus', async () => {
    const prepared = await prepare(configFor(0), { corpus: fakeCorpus });
    expect(prepared.tokens.length).toBeGreaterThan(100);
    expect(prepared.vocabulary).toContain('cat');
  });

  it('rejects a corpus that produces no tokens', async () => {
    await expect(prepare({ ...configFor(0), corpus: 'empty' }, { corpus: fakeCorpus })).rejects.toThrow();
  });

  it('splits into training and held-out tokens', async () => {
    const config = configFor(1);
    const prepared = await prepare(config, { corpus: fakeCorpus });
    const state = initState(config, rulesFor(1), prepared);

    expect(state.heldOutTokens.length).toBeGreaterThan(0);
    expect(state.trainTokens.length + state.heldOutTokens.length).toBe(prepared.tokens.length);
  });

  it('builds the table from the training portion only', async () => {
    const config = configFor(1);
    const prepared = await prepare(config, { corpus: fakeCorpus });
    const state = initState(config, rulesFor(1), prepared);
    expect(state.table.n).toBe(config.n);
    expect(state.table.counts.size).toBeGreaterThan(0);
  });
});

describe('ngramPredictionEngine — beat the model', () => {
  const config = configFor(0);

  it('builds rounds whose true answer is among the candidates', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    const state = initState(config, rulesFor(0), prepared);

    expect(state.rounds.length).toBeGreaterThan(0);
    for (const round of state.rounds) {
      expect(round.candidates).toContain(round.trueNext);
      expect(round.candidates.length).toBeLessThanOrEqual(config.candidateCount!);
      expect(round.answer).toBeNull();
    }
  });

  it('scores all-correct answers at 1', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    let state = initState(config, rulesFor(0), prepared);
    state.rounds.forEach((round, i) => {
      state = applyAction(state, { type: 'ANSWER', roundIndex: i, word: round.trueNext });
    });

    const result = evaluate(state);
    expect(result.metric).toBe('predictionAccuracy');
    expect(result.value).toBeCloseTo(1);
    expect(result.stars).toBe(3);
  });

  it('rejects a word that is not a candidate', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    let state = initState(config, rulesFor(0), prepared);
    state = applyAction(state, { type: 'ANSWER', roundIndex: 0, word: 'not-a-candidate' });
    expect(state.rounds[0]!.answer).toBeNull();
  });

  it('counts unanswered rounds as wrong', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    let state = initState(config, rulesFor(0), prepared);
    state = applyAction(state, { type: 'ANSWER', roundIndex: 0, word: state.rounds[0]!.trueNext });
    expect(evaluate(state).value).toBeCloseTo(1 / state.rounds.length);
  });

  it('surfaces the table top predictions for a context', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    const state = initState(config, rulesFor(0), prepared);
    const predictions = topPredictions(state, ['the'], 3);

    expect(predictions.length).toBeGreaterThan(0);
    expect(predictions[0]!.probability).toBeGreaterThanOrEqual(predictions.at(-1)!.probability);
    // "cat" follows "the" more than anything else in this corpus.
    expect(predictions[0]!.word).toBe('cat');
  });
});

describe('ngramPredictionEngine — tune order', () => {
  const config = configFor(1);

  it('rebuilds the table when the order changes', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    let state = initState(config, rulesFor(1), prepared);
    state = applyAction(state, { type: 'SET_N', value: 3 });

    expect(state.n).toBe(3);
    expect(state.table.n).toBe(3);
  });

  it('clamps the order to its configured range', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    let state = initState(config, rulesFor(1), prepared);
    state = applyAction(state, { type: 'SET_N', value: 99 });
    expect(state.n).toBe(config.nRange![1]);
    state = applyAction(state, { type: 'SET_N', value: -5 });
    expect(state.n).toBe(config.nRange![0]);
  });

  it('clamps smoothing alpha to its range', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    let state = initState(config, rulesFor(1), prepared);
    state = applyAction(state, { type: 'SET_ALPHA', value: 99 });
    expect(state.alpha).toBe(config.smoothingAlphaRange![1]);
  });

  it('improves perplexity going from unigram to bigram on structured text', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    let unigram = initState(config, rulesFor(1), prepared);
    unigram = applyAction(unigram, { type: 'SET_N', value: 1 });
    let bigram = initState(config, rulesFor(1), prepared);
    bigram = applyAction(bigram, { type: 'SET_N', value: 2 });

    expect(evaluate(bigram).value).toBeLessThan(evaluate(unigram).value);
  });

  it('reports coverage alongside perplexity', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    const state = initState(config, rulesFor(1), prepared);
    const result = evaluate(state);
    expect(result.metric).toBe('heldOutPerplexity');
    expect(result.breakdown.coverage).toBeGreaterThanOrEqual(0);
    expect(result.breakdown.coverage).toBeLessThanOrEqual(1);
  });
});

describe('ngramPredictionEngine — sparsity wall', () => {
  const config = configFor(2);

  it('collapses coverage as the order climbs', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    let low = initState(config, rulesFor(2), prepared);
    low = applyAction(low, { type: 'SET_N', value: 2 });
    let high = initState(config, rulesFor(2), prepared);
    high = applyAction(high, { type: 'SET_N', value: 7 });

    expect(evaluate(high).breakdown.coverage).toBeLessThan(evaluate(low).breakdown.coverage!);
  });

  it('awards nothing once coverage falls below the floor', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    let state = initState(config, rulesFor(2), prepared);
    state = applyAction(state, { type: 'SET_N', value: config.nRange![1] });

    const result = evaluate(state);
    if (result.breakdown.coverage! < config.minCoverage!) {
      expect(result.value).toBe(0);
    }
  });

  it('rewards a higher order while coverage still holds', async () => {
    const prepared = await prepare(config, { corpus: fakeCorpus });
    const scores = [2, 3, 4, 5, 6, 7].map((n) => {
      let state = initState(config, rulesFor(2), prepared);
      state = applyAction(state, { type: 'SET_N', value: n });
      const result = evaluate(state);
      return { n, value: result.value, coverage: result.breakdown.coverage! };
    });

    const viable = scores.filter((s) => s.coverage >= config.minCoverage!);
    expect(viable.length).toBeGreaterThan(0);
    const best = viable.reduce((a, b) => (b.value > a.value ? b : a));
    expect(best.n).toBe(Math.max(...viable.map((s) => s.n)));
  });
});

describe('ngramPredictionEngine — level config coverage', () => {
  it('handles every shipped level', async () => {
    for (const [i, level] of game.levels.entries()) {
      const config = level.engineConfig as unknown as NgramPredictionConfig;
      const prepared = await prepare(config, { corpus: fakeCorpus });
      const state = initState(config, rulesFor(i), prepared);
      const result = evaluate(state);
      expect(result.metric).toBe(level.passCriteria.metric);
    }
  });
});

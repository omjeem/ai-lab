import { describe, it, expect } from 'vitest';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  type SimilarityRankConfig,
} from '@/engines/similarityRankEngine';
import type { Embedder } from '@/engines/deps';
import type { EngineRules } from '@/types/game';
import game from '@data/games/world-1-fundamentals/1-3-similarity-distance.json';

const SPACE: Record<string, number[]> = {
  doctor: [1, 0, 0],
  surgeon: [0.95, 0.1, 0],
  nurse: [0.8, 0.3, 0],
  hospital: [0.5, 0.6, 0],
  lawyer: [0.2, 0.9, 0],
  bicycle: [0, 0, 1],
};

const fakeEmbedder: Embedder = {
  async embed(texts) {
    return texts.map((t) => SPACE[t] ?? [((t.charCodeAt(0) % 5) + 1) / 5, (t.length % 4) / 4, 0.2]);
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
const configFor = (i: number) => game.levels[i]!.engineConfig as unknown as SimilarityRankConfig;

describe('similarityRankEngine — rank mode', () => {
  const config = configFor(0);

  it('computes the true ordering from live similarities to the anchor', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    const state = initState(config, rulesFor(0), prepared);

    expect(state.trueOrder[0]).toBe('surgeon');
    expect(state.trueOrder.at(-1)).toBe('bicycle');
    expect(state.trueOrder).toHaveLength(config.words!.length);
  });

  it('starts with the ordering as configured, not pre-solved', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    const state = initState(config, rulesFor(0), prepared);
    expect(state.ordering).toEqual(config.words);
    expect(state.status).toBe('idle');
  });

  it('scores a perfect ordering at 1', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(0), prepared);
    state = applyAction(state, { type: 'SET_ORDER', ordering: [...state.trueOrder] });

    const result = evaluate(state);
    expect(result.metric).toBe('rankCorrelation');
    expect(result.value).toBeCloseTo(1);
    expect(result.stars).toBe(3);
  });

  it('scores a fully reversed ordering at -1 and fails it', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(0), prepared);
    state = applyAction(state, { type: 'SET_ORDER', ordering: [...state.trueOrder].reverse() });

    const result = evaluate(state);
    expect(result.value).toBeCloseTo(-1);
    expect(result.passed).toBe(false);
    expect(result.stars).toBe(0);
  });

  it('moves a single item and keeps every word exactly once', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(0), prepared);
    state = applyAction(state, { type: 'MOVE_ITEM', from: 0, to: 3 });

    expect(new Set(state.ordering).size).toBe(config.words!.length);
    expect(state.ordering).toHaveLength(config.words!.length);
    expect(state.ordering[3]).toBe(config.words![0]);
  });

  it('ignores a move with out-of-range indices', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    const state = initState(config, rulesFor(0), prepared);
    expect(applyAction(state, { type: 'MOVE_ITEM', from: -1, to: 2 }).ordering).toEqual(state.ordering);
    expect(applyAction(state, { type: 'MOVE_ITEM', from: 0, to: 99 }).ordering).toEqual(state.ordering);
  });

  it('rejects an ordering that is not a permutation of the words', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    const state = initState(config, rulesFor(0), prepared);
    const next = applyAction(state, { type: 'SET_ORDER', ordering: ['doctor', 'doctor'] });
    expect(next.ordering).toEqual(state.ordering);
  });
});

describe('similarityRankEngine — odd one out', () => {
  const config = configFor(1);

  it('derives the outlier from mean pairwise similarity, live', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    const state = initState(config, rulesFor(1), prepared);

    expect(state.sets).toHaveLength(config.sets!.length);
    for (const set of state.sets) {
      expect(set.words).toContain(set.trueOutlier);
      expect(set.answer).toBeNull();
    }
  });

  it('awards full accuracy when every outlier is found', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(1), prepared);

    state.sets.forEach((set, i) => {
      state = applyAction(state, { type: 'ANSWER_ODD', setIndex: i, word: set.trueOutlier });
    });

    const result = evaluate(state);
    expect(result.metric).toBe('oddOneOutAccuracy');
    expect(result.value).toBeCloseTo(1);
    expect(result.stars).toBe(3);
  });

  it('gives partial credit and can fail below the threshold', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(1), prepared);

    state.sets.forEach((set, i) => {
      const wrong = set.words.find((w) => w !== set.trueOutlier)!;
      state = applyAction(state, { type: 'ANSWER_ODD', setIndex: i, word: wrong });
    });

    const result = evaluate(state);
    expect(result.value).toBe(0);
    expect(result.passed).toBe(false);
  });

  it('rejects an answer that is not in that set', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(1), prepared);
    state = applyAction(state, { type: 'ANSWER_ODD', setIndex: 0, word: 'not-present' });
    expect(state.sets[0]!.answer).toBeNull();
  });

  it('ignores an out-of-range set index', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    const state = initState(config, rulesFor(1), prepared);
    expect(applyAction(state, { type: 'ANSWER_ODD', setIndex: 42, word: 'piano' }).sets).toEqual(state.sets);
  });

  it('counts unanswered sets as wrong', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(1), prepared);
    state = applyAction(state, { type: 'ANSWER_ODD', setIndex: 0, word: state.sets[0]!.trueOutlier });
    expect(evaluate(state).value).toBeCloseTo(1 / state.sets.length);
  });
});

describe('similarityRankEngine — free set', () => {
  const config = configFor(2);

  it('starts empty and generates no questions before the minimum word count', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    const state = initState(config, rulesFor(2), prepared);
    expect(state.words).toEqual([]);
    expect(state.questions).toEqual([]);
  });

  it('generates disagreement questions once enough words are added', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(2), prepared);

    for (const word of ['doctor', 'surgeon', 'nurse', 'hospital', 'lawyer']) {
      state = applyAction(state, { type: 'ADD_WORD', word, vector: SPACE[word]! });
    }

    expect(state.words).toHaveLength(5);
    expect(state.questions).toHaveLength(config.disagreementRounds!);
    for (const q of state.questions) {
      expect(state.words).toContain(q.anchor);
      expect(typeof q.trueDisagreement).toBe('boolean');
      expect(q.answer).toBeNull();
    }
  });

  it('refuses to exceed maxWords and refuses duplicates', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(2), prepared);

    for (let i = 0; i < config.maxWords! + 5; i++) {
      state = applyAction(state, { type: 'ADD_WORD', word: `w${i}`, vector: [i, i % 3, 1] });
    }
    expect(state.words).toHaveLength(config.maxWords!);

    const before = state.words.length;
    state = applyAction(state, { type: 'ADD_WORD', word: 'w0', vector: [0, 0, 1] });
    expect(state.words).toHaveLength(before);
  });

  it('scores every correct disagreement call at 1', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(2), prepared);
    for (const word of ['doctor', 'surgeon', 'nurse', 'hospital', 'lawyer']) {
      state = applyAction(state, { type: 'ADD_WORD', word, vector: SPACE[word]! });
    }

    state.questions.forEach((q, i) => {
      state = applyAction(state, { type: 'ANSWER_DISAGREEMENT', questionIndex: i, value: q.trueDisagreement });
    });

    const result = evaluate(state);
    expect(result.metric).toBe('metricDisagreementAccuracy');
    expect(result.value).toBeCloseTo(1);
    expect(result.stars).toBe(3);
  });

  it('scores every wrong call at 0', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(2), prepared);
    for (const word of ['doctor', 'surgeon', 'nurse', 'hospital', 'lawyer']) {
      state = applyAction(state, { type: 'ADD_WORD', word, vector: SPACE[word]! });
    }

    state.questions.forEach((q, i) => {
      state = applyAction(state, { type: 'ANSWER_DISAGREEMENT', questionIndex: i, value: !q.trueDisagreement });
    });

    expect(evaluate(state).value).toBe(0);
  });

  it('scores zero when no questions exist yet', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    expect(evaluate(initState(config, rulesFor(2), prepared)).value).toBe(0);
  });
});

describe('similarityRankEngine — level config coverage', () => {
  it('handles every shipped level end to end', async () => {
    for (const [i, level] of game.levels.entries()) {
      const config = level.engineConfig as unknown as SimilarityRankConfig;
      const prepared = await prepare(config, { embedder: fakeEmbedder });
      const state = initState(config, rulesFor(i), prepared);
      const result = evaluate(state);
      expect(result.metric).toBe(level.passCriteria.metric);
      expect(Number.isFinite(result.value)).toBe(true);
    }
  });
});

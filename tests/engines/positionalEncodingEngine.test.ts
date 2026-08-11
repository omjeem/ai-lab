import { describe, it, expect } from 'vitest';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  positionalVector,
  encodingDistinctness,
  combineWithPositions,
  type PositionalEncodingConfig,
} from '@/engines/positionalEncodingEngine';
import type { Embedder } from '@/engines/deps';
import type { EngineRules } from '@/types/game';
import { cosineSimilarity } from '@/engines/shared';
import game from '@data/games/world-5-transformers/5-1-positional-encoding.json';

const fakeEmbedder: Embedder = {
  async embed(texts) {
    return texts.map((t) => {
      const seed = t.charCodeAt(0) + t.length;
      return Array.from({ length: 16 }, (_, i) => Math.sin(seed * (i + 1) * 0.37));
    });
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
const configFor = (i: number) =>
  game.levels[i]!.engineConfig as unknown as PositionalEncodingConfig;

describe('positionalVector', () => {
  it('returns one value per dimension, alternating sine and cosine', () => {
    const v = positionalVector(3, 8, 10000);
    expect(v).toHaveLength(8);
    expect(v[0]).toBeCloseTo(Math.sin(3));
    expect(v[1]).toBeCloseTo(Math.cos(3));
  });

  it('keeps every value inside the sinusoid range', () => {
    for (const value of positionalVector(17, 64, 10000)) {
      expect(value).toBeGreaterThanOrEqual(-1);
      expect(value).toBeLessThanOrEqual(1);
    }
  });

  it('gives position zero a fixed signature', () => {
    const v = positionalVector(0, 8, 10000);
    expect(v.filter((_, i) => i % 2 === 0).every((x) => Math.abs(x) < 1e-12)).toBe(true);
    expect(v.filter((_, i) => i % 2 === 1).every((x) => Math.abs(x - 1) < 1e-12)).toBe(true);
  });

  it('makes nearby positions more similar than distant ones', () => {
    const a = positionalVector(4, 64, 10000);
    const near = positionalVector(5, 64, 10000);
    const far = positionalVector(40, 64, 10000);
    expect(cosineSimilarity(a, near)).toBeGreaterThan(cosineSimilarity(a, far));
  });

  it('extends to positions beyond any training length without special-casing', () => {
    expect(positionalVector(100_000, 16, 10000).every(Number.isFinite)).toBe(true);
  });
});

describe('encodingDistinctness', () => {
  it('improves with more dimensions only when the base spreads the frequencies', () => {
    // With a small base the extra dimensions carry genuinely different
    // frequencies and separability rises.
    expect(encodingDistinctness(64, 64, 10)).toBeGreaterThan(encodingDistinctness(64, 4, 10));

    // With a large base the added dimensions oscillate too slowly to vary over
    // this many positions, so they pull every vector towards the same shape.
    expect(encodingDistinctness(64, 64, 100_000)).toBeLessThan(
      encodingDistinctness(64, 4, 100_000)
    );
  });

  it('falls as the base grows past the sequence it has to cover', () => {
    expect(encodingDistinctness(64, 32, 10)).toBeGreaterThan(encodingDistinctness(64, 32, 100_000));
  });

  it('stays inside the unit range', () => {
    for (const d of [4, 16, 64]) {
      const value = encodingDistinctness(16, d, 10000);
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(1);
    }
  });

  it('is perfectly distinct with a single position', () => {
    expect(encodingDistinctness(1, 8, 10000)).toBe(1);
  });
});

describe('positionalEncodingEngine — identify position', () => {
  const config = configFor(0);

  it('builds rounds whose answer is among the candidates', () => {
    const state = initState(config, rulesFor(0));
    expect(state.rounds).toHaveLength(config.rounds!);
    for (const round of state.rounds) {
      expect(round.candidates).toContain(round.truePosition);
      expect(round.encoding).toHaveLength(config.dModel);
      expect(round.answer).toBeNull();
    }
  });

  it('is deterministic for the configured seed', () => {
    const a = initState(config, rulesFor(0));
    const b = initState(config, rulesFor(0));
    expect(a.rounds.map((r) => r.truePosition)).toEqual(b.rounds.map((r) => r.truePosition));
  });

  it('scores all-correct answers at 1', () => {
    let state = initState(config, rulesFor(0));
    state.rounds.forEach((round, i) => {
      state = applyAction(state, { type: 'ANSWER_POSITION', roundIndex: i, value: round.truePosition });
    });

    const result = evaluate(state);
    expect(result.metric).toBe('positionIdentificationAccuracy');
    expect(result.value).toBeCloseTo(1);
    expect(result.stars).toBe(3);
  });

  it('rejects an answer outside the candidates', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, { type: 'ANSWER_POSITION', roundIndex: 0, value: 9999 });
    expect(state.rounds[0]!.answer).toBeNull();
  });

  it('counts unanswered rounds as wrong', () => {
    let state = initState(config, rulesFor(0));
    state = applyAction(state, {
      type: 'ANSWER_POSITION',
      roundIndex: 0,
      value: state.rounds[0]!.truePosition,
    });
    expect(evaluate(state).value).toBeCloseTo(1 / config.rounds!);
  });
});

describe('positionalEncodingEngine — tune distinctness', () => {
  const config = configFor(1);

  it('starts below the pass threshold at the configured settings', () => {
    const state = initState(config, rulesFor(1));
    expect(evaluate(state).passed).toBe(false);
  });

  it('can be tuned above the pass threshold', () => {
    let best = 0;
    for (const dModel of config.dModelOptions!) {
      for (const base of [10, 100, 1000, 10000, 100000]) {
        let state = initState(config, rulesFor(1));
        state = applyAction(state, { type: 'SET_D_MODEL', value: dModel });
        state = applyAction(state, { type: 'SET_BASE', value: base });
        best = Math.max(best, evaluate(state).value);
      }
    }
    expect(best).toBeGreaterThanOrEqual(game.levels[1]!.passCriteria.threshold);
  });

  it('refuses a dimension count that was not offered', () => {
    let state = initState(config, rulesFor(1));
    const before = state.dModel;
    state = applyAction(state, { type: 'SET_D_MODEL', value: 7 });
    expect(state.dModel).toBe(before);
  });

  it('clamps the base to its range and rejects degenerate values', () => {
    let state = initState(config, rulesFor(1));
    state = applyAction(state, { type: 'SET_BASE', value: 1e9 });
    expect(state.base).toBe(config.baseRange![1]);
    const before = state.base;
    state = applyAction(state, { type: 'SET_BASE', value: 0.5 });
    expect(state.base).toBe(before);
  });
});

describe('positionalEncodingEngine — combine with embeddings', () => {
  const config = configFor(2);

  it('embeds every word across both sentences of every pair', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    expect(Object.keys(prepared.vectors).length).toBeGreaterThan(0);
    expect(prepared.vectors['dog']).toBeDefined();
    expect(prepared.vectors['man']).toBeDefined();
  });

  it('returns no vectors for the pure-maths levels', async () => {
    const prepared = await prepare(configFor(0), { embedder: fakeEmbedder });
    expect(prepared.vectors).toEqual({});
  });

  it('measures a real divergence between reordered sentences', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    const state = initState(config, rulesFor(2), prepared);

    expect(state.pairs).toHaveLength(config.sentencePairs!.length);
    for (const pair of state.pairs) {
      expect(pair.trueDivergence).toBeGreaterThan(0);
      expect(pair.trueDivergence).toBeLessThanOrEqual(1);
    }
  });

  it('collapses divergence to zero when the encoding is scaled out', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(2), prepared);
    state = applyAction(state, { type: 'SET_SCALE', value: 0 });

    // With no positional component a word is the same vector wherever it sits.
    for (const pair of state.pairs) {
      expect(pair.trueDivergence).toBeCloseTo(0, 6);
    }
  });

  it('grows divergence as the encoding is scaled in', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let low = initState(config, rulesFor(2), prepared);
    low = applyAction(low, { type: 'SET_SCALE', value: 0.2 });
    let high = initState(config, rulesFor(2), prepared);
    high = applyAction(high, { type: 'SET_SCALE', value: 2 });

    expect(high.pairs[0]!.trueDivergence).toBeGreaterThan(low.pairs[0]!.trueDivergence);
  });

  it('scores exact estimates at zero error', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(2), prepared);
    state.pairs.forEach((pair, i) => {
      state = applyAction(state, {
        type: 'ESTIMATE_DIVERGENCE',
        pairIndex: i,
        value: pair.trueDivergence,
      });
    });

    const result = evaluate(state);
    expect(result.metric).toBe('divergenceEstimateError');
    expect(result.value).toBeCloseTo(0);
    expect(result.stars).toBe(3);
  });

  it('charges maximum error for unanswered pairs', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    const state = initState(config, rulesFor(2), prepared);
    expect(evaluate(state).value).toBe(1);
  });

  it('emits one vector per known word, keeping the sequence intact', () => {
    const combined = combineWithPositions(['known', 'missing'], { known: [1, 0, 0, 0] }, 10000, 1);
    expect(combined).toHaveLength(1);
    expect(combined[0]).toHaveLength(4);
    expect(combined[0]!.every(Number.isFinite)).toBe(true);
  });

  it('returns an empty vector when no word is known', () => {
    expect(combineWithPositions(['a', 'b'], {}, 10000, 1)).toEqual([]);
  });
});

describe('positionalEncodingEngine — level config coverage', () => {
  it('handles every shipped level', async () => {
    for (const [i, level] of game.levels.entries()) {
      const config = level.engineConfig as unknown as PositionalEncodingConfig;
      const prepared = await prepare(config, { embedder: fakeEmbedder });
      const state = initState(config, rulesFor(i), prepared);
      const result = evaluate(state);
      expect(result.metric).toBe(level.passCriteria.metric);
      expect(Number.isFinite(result.value)).toBe(true);
    }
  });
});

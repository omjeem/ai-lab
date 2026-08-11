import { describe, it, expect } from 'vitest';
import {
  prepare,
  initState,
  applyAction,
  evaluate,
  type VectorArithmeticConfig,
} from '@/engines/vectorArithmeticEngine';
import type { Embedder } from '@/engines/deps';
import type { EngineRules } from '@/types/game';
import game from '@data/games/world-1-fundamentals/1-2-vector-arithmetic.json';

/**
 * Planted embedding space where the analogy actually resolves:
 * dimension 0 is royalty, 1 is gender, 2 is a per-word tag. king - man + woman
 * lands exactly on queen.
 */
const SPACE: Record<string, number[]> = {
  king: [1, 1, 0],
  man: [0, 1, 0],
  woman: [0, 0, 0],
  queen: [1, 0, 0],
  prince: [0.9, 1, 0.2],
  throne: [0.7, 0, 0.6],
  castle: [0.5, 0, 0.9],
  soldier: [0.2, 0.8, 0.5],
  farmer: [0.1, 0.5, 0.7],
  river: [0, 0, 1],
  engine: [0, 0.1, 1.1],
};

const fakeEmbedder: Embedder = {
  async embed(texts: string[]) {
    return texts.map((t) => SPACE[t] ?? [t.length / 10, (t.charCodeAt(0) % 7) / 10, 0.5]);
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
const configFor = (i: number) => game.levels[i]!.engineConfig as unknown as VectorArithmeticConfig;

describe('vectorArithmeticEngine — prepare', () => {
  it('embeds every term and candidate exactly once', async () => {
    const seen: string[][] = [];
    const spy: Embedder = {
      async embed(texts) {
        seen.push(texts);
        return fakeEmbedder.embed(texts);
      },
    };
    const config = configFor(0);
    const prepared = await prepare(config, { embedder: spy });

    for (const c of config.candidatePool) expect(prepared.vectors[c]).toBeDefined();
    for (const t of config.terms ?? []) expect(prepared.vectors[t.word]).toBeDefined();
    // One batched call — a per-word call per candidate would be wasteful.
    expect(seen).toHaveLength(1);
    expect(new Set(seen[0]).size).toBe(seen[0]!.length);
  });

  it('includes the target word for free-term levels', async () => {
    const prepared = await prepare(configFor(1), { embedder: fakeEmbedder });
    expect(prepared.vectors['paris']).toBeDefined();
  });

  it('throws when the embedder returns a mismatched batch', async () => {
    const broken: Embedder = { async embed() { return []; } };
    await expect(prepare(configFor(0), { embedder: broken })).rejects.toThrow();
  });
});

describe('vectorArithmeticEngine — fixed analogy', () => {
  it('computes the result vector live and ranks the real nearest neighbour first', async () => {
    const config = configFor(0);
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    const state = initState(config, rulesFor(0), prepared);

    expect(state.resultVector).toEqual([1, 0, 0]);
    expect(state.ranked[0]!.word).toBe('queen');
    expect(state.ranked[0]!.similarity).toBeCloseTo(1);
    expect(state.ranked).toHaveLength(config.candidatePool.length);
  });

  it('starts unrevealed and idle', async () => {
    const config = configFor(0);
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    const state = initState(config, rulesFor(0), prepared);
    expect(state.revealed).toBe(false);
    expect(state.status).toBe('idle');
    expect(state.guess).toBeNull();
  });

  it('scores a top-1 guess at 3 and a third-place guess at 1', async () => {
    const config = configFor(0);
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(0), prepared);

    state = applyAction(state, { type: 'GUESS', word: state.ranked[0]!.word });
    expect(evaluate(state).value).toBe(3);
    expect(evaluate(state).stars).toBe(3);

    let third = initState(config, rulesFor(0), prepared);
    third = applyAction(third, { type: 'GUESS', word: third.ranked[2]!.word });
    const result = evaluate(third);
    expect(result.value).toBe(1);
    expect(result.passed).toBe(true);
    expect(result.stars).toBe(1);
  });

  it('scores a guess outside the top three at zero', async () => {
    const config = configFor(0);
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(0), prepared);
    state = applyAction(state, { type: 'GUESS', word: state.ranked.at(-1)!.word });

    const result = evaluate(state);
    expect(result.value).toBe(0);
    expect(result.passed).toBe(false);
    expect(result.xpEarned).toBe(0);
  });

  it('scores zero when no guess was made', async () => {
    const config = configFor(0);
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    expect(evaluate(initState(config, rulesFor(0), prepared)).value).toBe(0);
  });

  it('ignores a guess that is not in the candidate pool', async () => {
    const config = configFor(0);
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(0), prepared);
    state = applyAction(state, { type: 'GUESS', word: 'bicycle' });
    expect(state.guess).toBeNull();
  });

  it('locks the guess once revealed', async () => {
    const config = configFor(0);
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(0), prepared);
    state = applyAction(state, { type: 'GUESS', word: 'queen' });
    state = applyAction(state, { type: 'REVEAL' });
    state = applyAction(state, { type: 'GUESS', word: 'castle' });

    expect(state.revealed).toBe(true);
    expect(state.guess).toBe('queen');
  });

  it('reports the real similarity of the nearest neighbour in the breakdown', async () => {
    const config = configFor(0);
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(0), prepared);
    state = applyAction(state, { type: 'GUESS', word: 'queen' });

    const result = evaluate(state);
    expect(result.breakdown.topSimilarity).toBeCloseTo(1);
    expect(result.breakdown.guessRank).toBe(1);
  });
});

describe('vectorArithmeticEngine — free terms', () => {
  const config = configFor(1);

  it('has no result until enough terms are set', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    const state = initState(config, rulesFor(1), prepared);
    expect(state.resultVector).toBeNull();
    expect(state.ranked).toEqual([]);
  });

  it('recomputes the ranking each time a term changes', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(1), prepared);

    state = applyAction(state, { type: 'SET_TERM', index: 0, word: 'france', vector: [1, 0, 0], op: 'add' });
    state = applyAction(state, { type: 'SET_TERM', index: 1, word: 'germany', vector: [0, 1, 0], op: 'subtract' });
    state = applyAction(state, { type: 'SET_TERM', index: 2, word: 'berlin', vector: [0, 1, 1], op: 'add' });

    expect(state.resultVector).toEqual([1, 0, 1]);
    expect(state.ranked.length).toBeGreaterThan(0);
  });

  it('rejects a term index outside the configured count', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(1), prepared);
    state = applyAction(state, { type: 'SET_TERM', index: 9, word: 'x', vector: [1, 0, 0], op: 'add' });
    expect(state.terms.every((t) => t.word !== 'x')).toBe(true);
  });

  it('scores 1 when the target ranks first and lower as it slips', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(1), prepared);

    const paris = prepared.vectors['paris']!;
    state = applyAction(state, { type: 'SET_TERM', index: 0, word: 'paris', vector: paris, op: 'add' });
    state = applyAction(state, { type: 'SET_TERM', index: 1, word: 'zero', vector: [0, 0, 0], op: 'add' });
    state = applyAction(state, { type: 'SET_TERM', index: 2, word: 'zero2', vector: [0, 0, 0], op: 'add' });
    state = applyAction(state, { type: 'COMMIT_ATTEMPT' });

    const result = evaluate(state);
    expect(result.metric).toBe('targetRankScore');
    expect(result.value).toBe(1);
    expect(result.stars).toBe(3);
  });

  it('keeps the best attempt rather than the last one', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(1), prepared);
    const paris = prepared.vectors['paris']!;

    state = applyAction(state, { type: 'SET_TERM', index: 0, word: 'paris', vector: paris, op: 'add' });
    state = applyAction(state, { type: 'SET_TERM', index: 1, word: 'z', vector: [0, 0, 0], op: 'add' });
    state = applyAction(state, { type: 'SET_TERM', index: 2, word: 'z2', vector: [0, 0, 0], op: 'add' });
    state = applyAction(state, { type: 'COMMIT_ATTEMPT' });
    const best = evaluate(state).value;

    state = applyAction(state, { type: 'SET_TERM', index: 0, word: 'engine', vector: SPACE['engine']!, op: 'add' });
    state = applyAction(state, { type: 'COMMIT_ATTEMPT' });

    expect(evaluate(state).value).toBe(best);
  });

  it('stops accepting attempts past maxAttempts', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(1), prepared);
    state = applyAction(state, { type: 'SET_TERM', index: 0, word: 'paris', vector: prepared.vectors['paris']!, op: 'add' });
    state = applyAction(state, { type: 'SET_TERM', index: 1, word: 'z', vector: [0, 0, 0], op: 'add' });
    state = applyAction(state, { type: 'SET_TERM', index: 2, word: 'z2', vector: [0, 0, 0], op: 'add' });

    for (let i = 0; i < config.maxAttempts! + 4; i++) {
      state = applyAction(state, { type: 'COMMIT_ATTEMPT' });
    }
    expect(state.attempts).toBe(config.maxAttempts);
  });

  it('scores zero before any attempt is committed', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    expect(evaluate(initState(config, rulesFor(1), prepared)).value).toBe(0);
  });
});

describe('vectorArithmeticEngine — estimate similarity', () => {
  const config = configFor(2);

  it('creates one round per configured round, each targeting a real candidate', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    const state = initState(config, rulesFor(2), prepared);

    expect(state.rounds).toHaveLength(config.rounds!);
    for (const round of state.rounds) {
      expect(config.candidatePool).toContain(round.word);
      expect(round.actualSimilarity).toBeGreaterThanOrEqual(-1);
      expect(round.actualSimilarity).toBeLessThanOrEqual(1);
      expect(round.estimate).toBeNull();
    }
  });

  it('scores a perfect estimate at zero error', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(2), prepared);

    state.rounds.forEach((round, i) => {
      state = applyAction(state, { type: 'ESTIMATE', roundIndex: i, value: round.actualSimilarity });
    });

    const result = evaluate(state);
    expect(result.metric).toBe('cosineSimilarityError');
    expect(result.value).toBeCloseTo(0);
    expect(result.passed).toBe(true);
    expect(result.stars).toBe(3);
  });

  it('grows the error with the size of the miss, after clamping', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(2), prepared);

    // Estimates are clamped into [-1, 1], so the realised error is the distance
    // to the clamped guess rather than the raw offset.
    const expected =
      state.rounds.reduce((sum, round) => {
        const guess = Math.min(1, round.actualSimilarity + 0.5);
        return sum + Math.abs(guess - round.actualSimilarity);
      }, 0) / state.rounds.length;

    state.rounds.forEach((round, i) => {
      state = applyAction(state, {
        type: 'ESTIMATE',
        roundIndex: i,
        value: round.actualSimilarity + 0.5,
      });
    });

    expect(evaluate(state).value).toBeCloseTo(expected);
    expect(expected).toBeGreaterThan(0);
  });

  it('fails a run whose estimates miss by more than the threshold', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(2), prepared);

    // Guess the opposite sign every time — a miss too large to clamp away.
    state.rounds.forEach((round, i) => {
      state = applyAction(state, { type: 'ESTIMATE', roundIndex: i, value: -round.actualSimilarity });
    });

    const result = evaluate(state);
    expect(result.value).toBeGreaterThan(game.levels[2]!.passCriteria.threshold);
    expect(result.passed).toBe(false);
    expect(result.stars).toBe(0);
  });

  it('clamps estimates into the valid cosine range', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(2), prepared);
    state = applyAction(state, { type: 'ESTIMATE', roundIndex: 0, value: 7 });
    expect(state.rounds[0]!.estimate).toBe(1);
    state = applyAction(state, { type: 'ESTIMATE', roundIndex: 0, value: -7 });
    expect(state.rounds[0]!.estimate).toBe(-1);
  });

  it('ignores an out-of-range round index', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    const state = initState(config, rulesFor(2), prepared);
    const next = applyAction(state, { type: 'ESTIMATE', roundIndex: 99, value: 0.5 });
    expect(next.rounds).toEqual(state.rounds);
  });

  it('charges maximum error for rounds left unanswered', async () => {
    const prepared = await prepare(config, { embedder: fakeEmbedder });
    let state = initState(config, rulesFor(2), prepared);
    state = applyAction(state, { type: 'ESTIMATE', roundIndex: 0, value: state.rounds[0]!.actualSimilarity });

    const result = evaluate(state);
    expect(result.value).toBeGreaterThan(0);
    expect(result.breakdown.answered).toBe(1);
  });
});

describe('vectorArithmeticEngine — level config coverage', () => {
  it('handles every shipped level end to end', async () => {
    for (const [i, level] of game.levels.entries()) {
      const config = level.engineConfig as unknown as VectorArithmeticConfig;
      const prepared = await prepare(config, { embedder: fakeEmbedder });
      const state = initState(config, rulesFor(i), prepared);
      const result = evaluate(state);
      expect(result.metric).toBe(level.passCriteria.metric);
      expect(Number.isFinite(result.value)).toBe(true);
    }
  });
});
